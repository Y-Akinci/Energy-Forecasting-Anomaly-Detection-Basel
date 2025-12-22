import os
import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

print("Index-TZ:", df.index.tz)
print(df.head())

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

df = df.loc[START_TS:END_TS].copy()
df_truth = df.copy()   # Rohdaten sichern für True-Werte

print("Zeitraum nach Filter:", df.index.min(), "→", df.index.max())
print("Anzahl Zeilen:", len(df))

# =======================================
# Targets für Multi-Output erzeugen
# =======================================

TARGET_COL = "Stromverbrauch"
HORIZON = 96  # 96 * 15min = 24h

for h in range(1, HORIZON + 1):
    df[f"{TARGET_COL}_t+{h}"] = df[TARGET_COL].shift(-h)

TARGET_COLS = [f"{TARGET_COL}_t+{h}" for h in range(1, HORIZON + 1)]

# Letzte Zeilen mit NaN in Targets entfernen
df = df.dropna(subset=TARGET_COLS)

print("Nach Target-Shift:")
print("Zeitraum:", df.index.min(), "→", df.index.max())
print("Anzahl Zeilen:", len(df))

# =======================================
# Feature-Setup
# =======================================

DROP_COLS = ["Datum (Lokal)", "Zeit (Lokal)"]

EXPERIMENT_NAME = "multi_output_24h"

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

WEATHER_FEATURES = [
    c for c in WEATHER_FEATURES_ALL
    if c not in EXCLUDE_WEATHER
]

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

print("\nAktive Features:", len(FEATURES))
print("Numerische:", len(NUMERIC_FEATURES))
print("Kategorische:", len(CATEGORICAL_FEATURES))

# =======================================
# Train / Test Split (zeitlich)
# =======================================

n = len(df)
split_idx = int(n * 0.7)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print("Train-Zeilen:", len(train_df), "| Zeitraum:",
      train_df.index.min(), "→", train_df.index.max())
print("Test-Zeilen :", len(test_df), "| Zeitraum:",
      test_df.index.min(), "→", test_df.index.max())

train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

y_train = train_df[TARGET_COLS].astype("float64")
y_test = test_df[TARGET_COLS].astype("float64")

print("X_train Shape:", X_train.shape, " | y_train:", y_train.shape)
print("X_test  Shape:", X_test.shape, " | y_test :", y_test.shape)

# =======================================
# Modelle (Multi-Output)
# =======================================

MODELS = {
    "XGBoost": MultiOutputRegressor(
        XGBRegressor(
            n_estimators=300,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.05,
            n_jobs=-1
        )
    )
}

results = {}

numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# =======================================
# Training + Evaluation
# =======================================

for name, model_obj in MODELS.items():
    print(f"\n=== Training {name} ===")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model_obj)
    ])

    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    # alles zusammen flatten für Gesamtmetriken über alle Horizonte
    yt_train = y_train.values.ravel()
    yp_train = y_pred_train.ravel()

    yt_test = y_test.values.ravel()
    yp_test = y_pred_test.ravel()

    metrics = {
        "MAE_train": mean_absolute_error(yt_train, yp_train),
        "RMSE_train": np.sqrt(mean_squared_error(yt_train, yp_train)),
        "R2_train": r2_score(yt_train, yp_train),

        "MAE_test": mean_absolute_error(yt_test, yp_test),
        "RMSE_test": np.sqrt(mean_squared_error(yt_test, yp_test)),
        "R2_test": r2_score(yt_test, yp_test),
    }

    results[name] = metrics

# Ergebnisse
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

results_df = pd.DataFrame(results).T

print("\n=== RESULTS (Train/Test, über alle 96 Horizonte gemittelt) ===")
print(results_df[[
    "MAE_train", "RMSE_train", "R2_train",
    "MAE_test", "RMSE_test", "R2_test"
]])

# =======================================
# Beispiel-Plot: nur Horizont t+1
# =======================================

pipe_final = pipe
y_pred_test = pipe_final.predict(X_test)

y_true_h1 = y_test.iloc[:, 0]      # t+1-Target
y_pred_h1 = y_pred_test[:, 0]      # t+1-Prediction

N = 500
plt.figure(figsize=(12, 4))
plt.plot(y_true_h1.index[:N], y_true_h1.values[:N], label="True t+1")
plt.plot(y_true_h1.index[:N], y_pred_h1[:N], label="Pred t+1")
plt.title("Test: True vs Pred (Horizont t+1)")
plt.legend()
plt.tight_layout()
plt.show()

# =======================================
# Hilfsfunktion: 24h-Multi-Output-Plot für eine Zeile
# =======================================

def plot_24h_from_row(row_pos, title_prefix="24h Multi-Output Forecast"):
    """
    Nimmt einen Row-Index in y_test und plottet True vs Pred
    fuer die naechsten 96 * 15min (24h).
    """
    start_ts = y_test.index[row_pos]  # Zeitpunkt t, zu dem Input vorliegt

    horizon_times = pd.date_range(
        start=start_ts + pd.Timedelta("15min"),
        periods=HORIZON,
        freq="15min",
        tz="UTC",
    )

    # True-Werte exakt auf den Horizonten t+1 ... t+96
    y_true_24h = df_truth.loc[horizon_times, TARGET_COL]

    # Vorhersage dieses Modells fuer diese Zeile
    y_pred_24h = y_pred_test[row_pos, :HORIZON]

    # Metriken fuer diesen einen 24h-Block
    mae_24h = mean_absolute_error(y_true_24h.values, y_pred_24h)
    rmse_24h = np.sqrt(mean_squared_error(y_true_24h.values, y_pred_24h))
    r2_24h = r2_score(y_true_24h.values, y_pred_24h)

    # In lokale Zeit fuer schoene Achse
    idx_local = y_true_24h.index.tz_convert("Europe/Zurich")

    plt.figure(figsize=(12, 4))
    plt.plot(idx_local, y_true_24h.values, label="True 24h")
    plt.plot(idx_local, y_pred_24h, label="Forecast 24h (Multi-Output)")
    plt.title(f"{title_prefix}\nMAE={mae_24h:.0f}, RMSE={rmse_24h:.0f}, R²={r2_24h:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Start t      : {start_ts}")
    print(f"Horizont von : {y_true_24h.index.min()} bis {y_true_24h.index.max()}")
    print(f"MAE 24h      : {mae_24h:.2f}")
    print(f"RMSE 24h     : {rmse_24h:.2f}")
    print(f"R² 24h       : {r2_24h:.4f}")


# =======================================
# 24h-Beispiel aus dem Testset (z.B. erste Zeile im Test)
# =======================================

plot_24h_from_row(0, title_prefix="24h Multi-Output Forecast (Beispieltag)")


# =======================================
# 24h Multi-Output Forecast fuer den 15.11.2024
# =======================================

target_start = pd.Timestamp("2024-11-15 00:00:00", tz="UTC")

if target_start not in y_test.index:
    print("Fehler: gewünschter Start-Timestamp nicht im Testset:", target_start)
else:
    row_pos_15 = y_test.index.get_loc(target_start)
    plot_24h_from_row(row_pos_15,
                      title_prefix="24h Multi-Output Forecast fuer 15.11.2024")
