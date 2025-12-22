import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# =======================================
# Einstellungen fuer Output
# =======================================

PRINT_DAILY_TABLE = False          # True: Tagesmetriken im Terminal anzeigen (nicht empfohlen)
SAVE_DAILY_CSV = True              # True: Tagesmetriken als CSV speichern
DAILY_CSV_DIRNAME = "reports"      # Ordner im Projektroot


# =======================================
# Pfade / Daten laden
# =======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_DIR = os.path.join(ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, "processed_merged_features.csv")

df = pd.read_csv(
    INPUT_CSV,
    sep=";",
    encoding="latin1",
    parse_dates=["Start der Messung (UTC)"]
)

df.set_index("Start der Messung (UTC)", inplace=True)

if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

df = df.loc[START_TS:END_TS].copy()
df_truth = df.copy()   # Rohdaten sichern fuer True-Werte (Stromverbrauch)

print("Index-TZ:", df.index.tz)
print("Zeitraum:", df.index.min(), "→", df.index.max(), "| Zeilen:", len(df))


# =======================================
# Targets fuer Multi-Output erzeugen
# =======================================

TARGET_COL = "Stromverbrauch"
HORIZON = 96  # 96 * 15min = 24h

for h in range(1, HORIZON + 1):
    df[f"{TARGET_COL}_t+{h}"] = df[TARGET_COL].shift(-h)

TARGET_COLS = [f"{TARGET_COL}_t+{h}" for h in range(1, HORIZON + 1)]
df = df.dropna(subset=TARGET_COLS)

print("Nach Target-Shift:", df.index.min(), "→", df.index.max(), "| Zeilen:", len(df))


# =======================================
# Feature-Setup
# =======================================

DROP_COLS = ["Datum (Lokal)", "Zeit (Lokal)"]

USE_WEATHER = True
USE_LAGS = True
USE_CALENDAR = True

EXCLUDE_WEATHER = [
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in km/h_lag15',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in m/s_lag15',
    'BÃ¶enspitze (SekundenbÃ¶e); Maximum in km/h_lag15',
    'Luftdruck reduziert auf Meeresniveau (QFF)_lag15',
    'Luftdruck reduziert auf Meeresniveau mit Standardatmosphäre (QNH)_lag15',
    'Luftrdruck auf Barometerhöhe_lag15',
    'Lufttemperatur 2 m ü. Gras_lag15',
    'Lufttemperatur Bodenoberfläche_lag15',
    'Windgeschwindigkeit vektoriell_lag15',
    'Windgeschwindigkeit; Zehnminutenmittel in km/h_lag15',
    'Windrichtung; Zehnminutenmittel_lag15',
    'relative Luftfeuchtigkeit_lag15'
]

EXCLUDE_LAGS = [
    "Grundversorgte Kunden_Lag_15min",
    "Freie Kunden_Lag_15min",
    "Lag_15min",
    "Lag_30min",
    "Diff_15min"
]

CALENDAR_FEATURES = [
    "Monat", "Wochentag", "Quartal",
    "Woche des Jahres", "Stunde (Lokal)",
    "IstArbeitstag", "IstSonntag"
]

LAG_FEATURES = [
    "Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h",
    "Grundversorgte Kunden_Lag_15min",
    "Freie Kunden_Lag_15min",
    "Diff_15min"
]

WEATHER_FEATURES_ALL = [
    c for c in df.columns
    if c.endswith("_lag15") and c not in LAG_FEATURES
]

WEATHER_FEATURES = [c for c in WEATHER_FEATURES_ALL if c not in EXCLUDE_WEATHER]
LAG_FEATURES = [c for c in LAG_FEATURES if c not in EXCLUDE_LAGS]

FEATURES = []
if USE_CALENDAR:
    FEATURES += CALENDAR_FEATURES
if USE_LAGS:
    FEATURES += LAG_FEATURES
if USE_WEATHER:
    FEATURES += WEATHER_FEATURES

FEATURES = [f for f in FEATURES if f in df.columns]

CATEGORICAL_FEATURES = [c for c in CALENDAR_FEATURES if c in FEATURES]
NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]

print("Aktive Features:", len(FEATURES), "| Num:", len(NUMERIC_FEATURES), "| Cat:", len(CATEGORICAL_FEATURES))


# =======================================
# Train / Test Split (zeitlich)
# =======================================

n = len(df)
split_idx = int(n * 0.7)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET_COLS].astype("float64")
y_test = test_df[TARGET_COLS].astype("float64")

print("Train:", train_df.index.min(), "→", train_df.index.max(), "|", len(train_df))
print("Test :", test_df.index.min(), "→", test_df.index.max(), "|", len(test_df))


# =======================================
# Modell (Multi-Output)
# =======================================

pipe = Pipeline([
    ("preprocess", ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])),
    ("model", MultiOutputRegressor(
        XGBRegressor(
            n_estimators=300,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.05,
            n_jobs=-1
        )
    ))
])

print("\n=== Training XGBoost Multi-Output ===")
pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)


# =======================================
# Gesamtmetriken als Tabelle
# =======================================

yt_train = y_train.values.ravel()
yp_train = y_pred_train.ravel()
yt_test = y_test.values.ravel()
yp_test = y_pred_test.ravel()

metrics_df = pd.DataFrame([{
    "MAE_train": mean_absolute_error(yt_train, yp_train),
    "RMSE_train": np.sqrt(mean_squared_error(yt_train, yp_train)),
    "R2_train": r2_score(yt_train, yp_train),
    "MAE_test": mean_absolute_error(yt_test, yp_test),
    "RMSE_test": np.sqrt(mean_squared_error(yt_test, yp_test)),
    "R2_test": r2_score(yt_test, yp_test),
}])

print("\n=== Gesamtmetriken (Multi-Output 24h, ueber alle 96 Horizonte geflattet) ===")
print(metrics_df.to_string(index=False))


# =======================================
# Tagesmetriken (nur erzeugen, nicht spam-printen)
# =======================================

def block_metrics(y_true_vals: np.ndarray, y_pred_vals: np.ndarray):
    mae = mean_absolute_error(y_true_vals, y_pred_vals)
    rmse = np.sqrt(mean_squared_error(y_true_vals, y_pred_vals))
    r2 = r2_score(y_true_vals, y_pred_vals)
    y_true_safe = np.where(y_true_vals == 0, 1e-6, y_true_vals)
    mape = np.mean(np.abs((y_true_vals - y_pred_vals) / y_true_safe)) * 100
    return mae, rmse, r2, mape

def daily_metrics_multioutput(y_df: pd.DataFrame, y_pred: np.ndarray, label: str):
    idx_local = y_df.index.tz_convert("Europe/Zurich")
    days = pd.Index(idx_local.date)

    rows = []
    for day in pd.unique(days):
        mask = (days == day)
        positions = np.where(mask)[0]
        if len(positions) == 0:
            continue

        row_pos = positions[0]

        y_true_24h = y_df.iloc[row_pos, :HORIZON].values
        y_pred_24h = y_pred[row_pos, :HORIZON]

        mae, rmse, r2, mape = block_metrics(y_true_24h, y_pred_24h)
        rows.append([pd.Timestamp(day), mae, rmse, r2, mape, y_df.index[row_pos]])

    out = pd.DataFrame(rows, columns=["Tag", "MAE_24h", "RMSE_24h", "R2_24h", "MAPE_24h_%", "Start_t(UTC)"]).sort_values("Tag")

    if PRINT_DAILY_TABLE:
        print(f"\n=== Tagesmetriken 24h Multi-Output ({label}) ===")
        print(out.to_string(index=False))

    if SAVE_DAILY_CSV:
        out_dir = os.path.join(ROOT, DAILY_CSV_DIRNAME)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"daily_metrics_multioutput_{label.lower()}.csv")
        out.to_csv(out_path, index=False, sep=";")
        print(f"Tagesmetriken gespeichert: {out_path}")

    return out

daily_train = daily_metrics_multioutput(y_train, y_pred_train, "Train")
daily_test = daily_metrics_multioutput(y_test, y_pred_test, "Test")


# =======================================
# Plot: Horizont t+1 (Quick-Check)
# =======================================

y_true_h1 = y_test.iloc[:, 0]
y_pred_h1 = y_pred_test[:, 0]

N = 500
plt.figure(figsize=(12, 4))
plt.plot(y_true_h1.index[:N], y_true_h1.values[:N], label="True t+1")
plt.plot(y_true_h1.index[:N], y_pred_h1[:N], label="Pred t+1")
plt.title("Test: True vs Pred (Horizont t+1)")
plt.legend()
plt.tight_layout()
plt.show()


# =======================================
# 24h Plot + Metriken fuer 15.11.2024
# =======================================

def plot_24h_for_timestamp(target_start_utc: pd.Timestamp, title_prefix="24h Multi-Output Forecast"):
    if target_start_utc not in y_test.index:
        print("Fehler: Start-Timestamp nicht im Testset:", target_start_utc)
        return

    row_pos = y_test.index.get_loc(target_start_utc)

    horizon_times = pd.date_range(
        start=target_start_utc + pd.Timedelta("15min"),
        periods=HORIZON,
        freq="15min",
        tz="UTC",
    )

    y_true_24h = df_truth.loc[horizon_times, TARGET_COL].values
    y_pred_24h = y_pred_test[row_pos, :HORIZON]

    mae_24h, rmse_24h, r2_24h, mape_24h = block_metrics(y_true_24h, y_pred_24h)

    idx_local = horizon_times.tz_convert("Europe/Zurich")

    plt.figure(figsize=(12, 4))
    plt.plot(idx_local, y_true_24h, label="True 24h")
    plt.plot(idx_local, y_pred_24h, label="Forecast 24h (Multi-Output)")
    plt.title(f"{title_prefix}\nMAE={mae_24h:.0f}, RMSE={rmse_24h:.0f}, R2={r2_24h:.3f}, MAPE={mape_24h:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n=== 24h Multi-Output Block ===")
    print("Start t (UTC):", target_start_utc)
    print(f"MAE 24h  : {mae_24h:.2f}")
    print(f"RMSE 24h : {rmse_24h:.2f}")
    print(f"R2 24h   : {r2_24h:.4f}")
    print(f"MAPE 24h : {mape_24h:.2f} %")

plot_24h_for_timestamp(pd.Timestamp("2024-11-15 00:00:00", tz="UTC"),
                       title_prefix="24h Multi-Output Forecast fuer 15.11.2024")

import joblib, os

MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_multioutput_24h_pipeline.joblib")
joblib.dump(pipe, MODEL_PATH)

print("Multi-Output Modell gespeichert:", MODEL_PATH)
print("EXISTS?:", os.path.isfile(MODEL_PATH))
print("ABS:", os.path.abspath(MODEL_PATH))
print("CWD:", os.getcwd())

