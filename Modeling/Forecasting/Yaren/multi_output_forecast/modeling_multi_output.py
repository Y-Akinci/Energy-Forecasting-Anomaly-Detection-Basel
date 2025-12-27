import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# -----------------------------
# Settings
# -----------------------------
SHOW_EXAMPLE_DAY_PLOT = True
EXAMPLE_DAY_UTC = pd.Timestamp("2024-11-15 00:00:00", tz="UTC")

HORIZON = 96
TARGET_COL = "Stromverbrauch"
TZ_LOCAL = "Europe/Zurich"

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# -----------------------------
# Helper
# -----------------------------
def metrics_1d(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)
    return float(mae), float(rmse), float(r2), float(mape)

def print_block(title: str, rows: list[tuple[str, float]]):
    print("\n" + title)
    for k, v in rows:
        if k.lower().startswith("r2"):
            print(f"{k:<10}: {v:.4f}")
        elif k.lower().startswith("mape"):
            print(f"{k:<10}: {v:.2f} %")
        else:
            print(f"{k:<10}: {v:.2f}")

# -----------------------------
# Load data
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR
while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
INPUT_CSV = DATA_DIR / "processed_merged_features.csv"

df = pd.read_csv(
    INPUT_CSV,
    sep=";",
    encoding="latin1",
    parse_dates=["Start der Messung (UTC)"]
)
df.set_index("Start der Messung (UTC)", inplace=True)
df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS   = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")
df = df.loc[START_TS:END_TS].copy()
df_truth = df.copy()

# -----------------------------
# Targets Multi-Output
# -----------------------------
target_cols = [f"{TARGET_COL}_t+{h}" for h in range(1, HORIZON + 1)]
targets = {col: df[TARGET_COL].shift(-h) for h, col in enumerate(target_cols, start=1)}
df_targets = pd.DataFrame(targets, index=df.index)
df = pd.concat([df, df_targets], axis=1).dropna(subset=target_cols)

# -----------------------------
# Features (wie bei dir)
# -----------------------------
DROP_COLS = ["Datum (Lokal)", "Zeit (Lokal)", "Monat", "Wochentag", "Stunde (Lokal)", "Tag des Jahres", "Quartal"]

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

df["Monat_sin"] = np.sin(2 * np.pi * df["Monat"] / 12)
df["Monat_cos"] = np.cos(2 * np.pi * df["Monat"] / 12)
df["Wochentag_sin"] = np.sin(2 * np.pi * df["Wochentag"] / 7)
df["Wochentag_cos"] = np.cos(2 * np.pi * df["Wochentag"] / 7)
df["Stunde_sin"] = np.sin(2 * np.pi * df["Stunde (Lokal)"] / 24)
df["Stunde_cos"] = np.cos(2 * np.pi * df["Stunde (Lokal)"] / 24)
df["TagJahr_sin"] = np.sin(2 * np.pi * df["Tag des Jahres"] / 365)
df["TagJahr_cos"] = np.cos(2 * np.pi * df["Tag des Jahres"] / 365)

CALENDAR_FEATURES = [
    "Monat_sin", "Monat_cos",
    "Wochentag_sin", "Wochentag_cos",
    "Stunde_sin", "Stunde_cos",
    "TagJahr_sin", "TagJahr_cos",
    "Woche des Jahres", "IstArbeitstag", "IstSonntag"
]

LAG_FEATURES_ALL = [
    "Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h",
    "Grundversorgte Kunden_Lag_15min",
    "Freie Kunden_Lag_15min",
    "Diff_15min"
]
LAG_FEATURES = [c for c in LAG_FEATURES_ALL if c not in EXCLUDE_LAGS]

WEATHER_FEATURES_ALL = [c for c in df.columns if c.endswith("_lag15") and c not in LAG_FEATURES_ALL]
WEATHER_FEATURES = [c for c in WEATHER_FEATURES_ALL if c not in EXCLUDE_WEATHER]

FEATURES = []
if USE_CALENDAR:
    FEATURES += CALENDAR_FEATURES
if USE_LAGS:
    FEATURES += LAG_FEATURES
if USE_WEATHER:
    FEATURES += WEATHER_FEATURES

NUMERIC_FEATURES = [f for f in FEATURES if f in df.columns]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# -----------------------------
# Train/Test Split
# -----------------------------
n = len(df)
split_idx = int(n * 0.7)
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

X_train = train_df[NUMERIC_FEATURES]
X_test  = test_df[NUMERIC_FEATURES]
y_train = train_df[target_cols].astype("float64")
y_test  = test_df[target_cols].astype("float64")

# -----------------------------
# Model + Train
# -----------------------------
preprocessor = ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES)], remainder="drop")

base_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
    force_col_wise=True,
)

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", MultiOutputRegressor(base_model, n_jobs=1))
])

pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test  = pipe.predict(X_test)

# -----------------------------
# 1) GLOBAL Train/Test (ueber alle Horizonte und alle Zeilen)
# -----------------------------
yt_train = y_train.values.ravel()
yp_train = y_pred_train.ravel()
yt_test  = y_test.values.ravel()
yp_test  = y_pred_test.ravel()

mae_tr, rmse_tr, r2_tr, mape_tr = metrics_1d(yt_train, yp_train)
mae_te, rmse_te, r2_te, mape_te = metrics_1d(yt_test, yp_test)

print_block("=== MULTI-OUTPUT 24h GLOBAL (Train) ===", [("MAE", mae_tr), ("RMSE", rmse_tr), ("R2", r2_tr), ("MAPE", mape_tr)])
print_block("=== MULTI-OUTPUT 24h GLOBAL (Test)  ===", [("MAE", mae_te), ("RMSE", rmse_te), ("R2", r2_te), ("MAPE", mape_te)])

# -----------------------------
# 2) 24h-Block-Auswertung (Start 00:00 lokal) Train/Test
#    Pro Startpunkt: 96 Werte -> Metriken, danach Mittelwert ueber alle Tage
# -----------------------------
def eval_midnight_blocks(y_df: pd.DataFrame, y_pred: np.ndarray, label: str):
    idx_local = y_df.index.tz_convert(TZ_LOCAL)
    mask_mid = (idx_local.hour == 0) & (idx_local.minute == 0)

    y_mid = y_df.loc[mask_mid]                   # DataFrame (rows = Starts)
    p_mid = y_pred[mask_mid, :]                  # np.array same rows

    if len(y_mid) == 0:
        print("\n=== MULTI-OUTPUT 24h BLOCK Start 00:00 lokal (" + label + ") ===")
        print("Keine Starts gefunden.")
        return None

    maes, rmses, r2s, mapes = [], [], [], []
    for i in range(len(y_mid)):
        y_true_96 = y_mid.iloc[i].values
        y_pred_96 = p_mid[i, :]
        mae, rmse, r2, mape = metrics_1d(y_true_96, y_pred_96)
        maes.append(mae); rmses.append(rmse); r2s.append(r2); mapes.append(mape)

    out = pd.DataFrame({"MAE": maes, "RMSE": rmses, "R2": r2s, "MAPE": mapes}, index=y_mid.index)

    print("\n=== MULTI-OUTPUT 24h BLOCK Start 00:00 lokal (" + label + ") ===")
    print("Tage ausgewertet:", len(out))
    print(f"MAE_mean  : {out['MAE'].mean():.2f} | RMSE_mean: {out['RMSE'].mean():.2f} | R2_mean: {out['R2'].mean():.4f} | MAPE_mean: {out['MAPE'].mean():.2f} %")
    print(f"MAE_median: {out['MAE'].median():.2f}")

    return out

train_mid_df = eval_midnight_blocks(y_train, y_pred_train, "Train")
test_mid_df  = eval_midnight_blocks(y_test,  y_pred_test,  "Test")

# -----------------------------
# 3) Optional: 1 Beispieltag Plot (Test)
# -----------------------------
if SHOW_EXAMPLE_DAY_PLOT and EXAMPLE_DAY_UTC in y_test.index:
    row_pos = y_test.index.get_loc(EXAMPLE_DAY_UTC)
    horizon_times = pd.date_range(EXAMPLE_DAY_UTC + pd.Timedelta("15min"), periods=HORIZON, freq="15min", tz="UTC")

    y_true_24h = df_truth.loc[horizon_times, TARGET_COL].values
    y_pred_24h = y_pred_test[row_pos, :]

    mae, rmse, r2, mape = metrics_1d(y_true_24h, y_pred_24h)
    idx_local = horizon_times.tz_convert(TZ_LOCAL)

    plt.figure(figsize=(12, 4))
    plt.plot(idx_local, y_true_24h, label="True 24h")
    plt.plot(idx_local, y_pred_24h, label="Forecast 24h (Multi-Output)")
    plt.title(f"Multi-Output 24h Beispieltag\nMAE={mae:.0f}, RMSE={rmse:.0f}, R2={r2:.3f}, MAPE={mape:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Save model
# -----------------------------
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
best_path = MODEL_DIR / "best_multioutput_24h_lgbm.joblib"
joblib.dump(pipe, best_path)
print("\nGespeichert:", best_path.resolve())
