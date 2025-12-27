import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# -----------------------------
# Settings
# -----------------------------
TARGET_COL = "Stromverbrauch"
HORIZON = 96
TZ_LOCAL = "Europe/Zurich"

SHOW_EXAMPLE_DAY_PLOT = True
EXAMPLE_DAY_UTC = pd.Timestamp("2024-11-15 00:00:00", tz="UTC")

MAX_MISSING_PER_BLOCK = 4
HISTORY_DAYS = 7

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

def _sin_cos(value, period):
    return np.sin(2 * np.pi * value / period), np.cos(2 * np.pi * value / period)

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
END_TS = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")
df = df.loc[START_TS:END_TS].copy()
df_truth = df.copy()

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
LAG_FEATURES = [
    "Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h",
    "Grundversorgte Kunden_Lag_15min",
    "Freie Kunden_Lag_15min",
    "Diff_15min"
]
WEATHER_FEATURES_ALL = [c for c in df.columns if c.endswith("_lag15") and c not in LAG_FEATURES]
WEATHER_FEATURES = [c for c in WEATHER_FEATURES_ALL if c not in EXCLUDE_WEATHER]
LAG_FEATURES = [c for c in LAG_FEATURES if c not in EXCLUDE_LAGS]

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
y_train = train_df[TARGET_COL].astype("float64")
X_test  = test_df[NUMERIC_FEATURES]
y_test  = test_df[TARGET_COL].astype("float64")

# -----------------------------
# 1-step model train
# -----------------------------
preprocessor = ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES)], remainder="drop")

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    ))
])

pipe.fit(X_train, y_train)

pred_train_1 = pipe.predict(X_train)
pred_test_1  = pipe.predict(X_test)

mae_tr, rmse_tr, r2_tr, mape_tr = metrics_1d(y_train.values, pred_train_1)
mae_te, rmse_te, r2_te, mape_te = metrics_1d(y_test.values,  pred_test_1)

print_block("=== REKURSIV Basis-Modell 1-step GLOBAL (Train) ===", [("MAE", mae_tr), ("RMSE", rmse_tr), ("R2", r2_tr), ("MAPE", mape_tr)])
print_block("=== REKURSIV Basis-Modell 1-step GLOBAL (Test)  ===", [("MAE", mae_te), ("RMSE", rmse_te), ("R2", r2_te), ("MAPE", mape_te)])

# -----------------------------
# Rekursiv Forecast Funktion
# -----------------------------
def forecast_multistep(pipe, df_history, df_full, steps=96):
    df_temp = df_history.copy().sort_index()
    preds = []

    for _ in range(steps):
        last_ts = df_temp.index[-1]
        next_ts = last_ts + pd.Timedelta("15min")

        row = {}

        # Kalender
        if "Monat_sin" in FEATURES or "Monat_cos" in FEATURES:
            s, c = _sin_cos(next_ts.month, 12)
            row["Monat_sin"] = s; row["Monat_cos"] = c

        if "Wochentag_sin" in FEATURES or "Wochentag_cos" in FEATURES:
            s, c = _sin_cos(next_ts.weekday(), 7)
            row["Wochentag_sin"] = s; row["Wochentag_cos"] = c

        if "Stunde_sin" in FEATURES or "Stunde_cos" in FEATURES:
            hour_local = next_ts.tz_convert(TZ_LOCAL).hour
            s, c = _sin_cos(hour_local, 24)
            row["Stunde_sin"] = s; row["Stunde_cos"] = c

        if "TagJahr_sin" in FEATURES or "TagJahr_cos" in FEATURES:
            doy = int(next_ts.tz_convert(TZ_LOCAL).dayofyear)
            s, c = _sin_cos(doy, 365)
            row["TagJahr_sin"] = s; row["TagJahr_cos"] = c

        if "Woche des Jahres" in FEATURES:
            row["Woche des Jahres"] = int(next_ts.isocalendar().week)
        if "IstArbeitstag" in FEATURES:
            row["IstArbeitstag"] = int(next_ts.weekday() < 5)
        if "IstSonntag" in FEATURES:
            row["IstSonntag"] = int(next_ts.weekday() == 6)

        # Lags Target
        if "Lag_15min" in FEATURES:
            row["Lag_15min"] = df_temp.iloc[-1][TARGET_COL]
        if "Lag_30min" in FEATURES and len(df_temp) >= 2:
            row["Lag_30min"] = df_temp.iloc[-2][TARGET_COL]
        if "Lag_1h" in FEATURES and len(df_temp) >= 4:
            row["Lag_1h"] = df_temp.iloc[-4][TARGET_COL]
        if "Lag_24h" in FEATURES and len(df_temp) >= 96:
            row["Lag_24h"] = df_temp.iloc[-96][TARGET_COL]

        if "Diff_15min" in FEATURES and len(df_temp) >= 2:
            row["Diff_15min"] = df_temp.iloc[-1][TARGET_COL] - df_temp.iloc[-2][TARGET_COL]

        # Kunden
        if "Grundversorgte Kunden_Lag_15min" in FEATURES and "Grundversorgte Kunden_Lag_15min" in df_temp.columns:
            row["Grundversorgte Kunden_Lag_15min"] = df_temp.iloc[-1]["Grundversorgte Kunden_Lag_15min"]
        if "Freie Kunden_Lag_15min" in FEATURES and "Freie Kunden_Lag_15min" in df_temp.columns:
            row["Freie Kunden_Lag_15min"] = df_temp.iloc[-1]["Freie Kunden_Lag_15min"]

        # Wetter
        if USE_WEATHER and len(WEATHER_FEATURES) > 0:
            if next_ts in df_full.index:
                for feat in WEATHER_FEATURES:
                    if feat in FEATURES:
                        row[feat] = df_full.loc[next_ts, feat]
            else:
                last_known_ts = df_full.index[df_full.index < next_ts].max()
                for feat in WEATHER_FEATURES:
                    if feat in FEATURES:
                        row[feat] = df_full.loc[last_known_ts, feat]

        X = pd.DataFrame([row], index=[next_ts])
        for c in NUMERIC_FEATURES:
            if c not in X.columns:
                X[c] = np.nan
        X = X[NUMERIC_FEATURES]

        y_hat = pipe.predict(X)[0]
        preds.append((next_ts, y_hat))

        df_temp.loc[next_ts] = {TARGET_COL: y_hat}

    return pd.DataFrame(preds, columns=["ts", "forecast"]).set_index("ts")

# -----------------------------
# 24h-Block-Auswertung 00:00 lokal (Train/Test)
# -----------------------------
def eval_recursive_midnight_blocks(starts: pd.DatetimeIndex, label: str):
    maes, rmses, r2s, mapes, kept = [], [], [], [], []

    for start_ts in starts:
        history_end = start_ts - pd.Timedelta("15min")
        history_start = history_end - pd.Timedelta(days=HISTORY_DAYS)
        history = df.loc[history_start:history_end].copy()
        if len(history) < 96:
            continue

        fc = forecast_multistep(pipe, history, df_full=df, steps=HORIZON)
        idx_block = fc.index

        truth = df.reindex(idx_block)[TARGET_COL]
        if int(truth.isna().sum()) > MAX_MISSING_PER_BLOCK:
            continue

        mask = truth.notna()
        y_true = truth[mask].values
        y_pred = fc.loc[truth.index[mask], "forecast"].values
        if len(y_true) < 10:
            continue

        mae, rmse, r2, mape = metrics_1d(y_true, y_pred)
        maes.append(mae); rmses.append(rmse); r2s.append(r2); mapes.append(mape)
        kept.append(start_ts)

    out = pd.DataFrame({"MAE": maes, "RMSE": rmses, "R2": r2s, "MAPE": mapes}, index=pd.DatetimeIndex(kept)).sort_index()

    print("\n=== REKURSIV 24h BLOCK Start 00:00 lokal (" + label + ") ===")
    print("Tage ausgewertet:", len(out))
    if len(out) > 0:
        print(f"MAE_mean  : {out['MAE'].mean():.2f} | RMSE_mean: {out['RMSE'].mean():.2f} | R2_mean: {out['R2'].mean():.4f} | MAPE_mean: {out['MAPE'].mean():.2f} %")
        print(f"MAE_median: {out['MAE'].median():.2f}")

    return out

train_local = train_df.index.tz_convert(TZ_LOCAL)
test_local  = test_df.index.tz_convert(TZ_LOCAL)

train_starts = train_df.index[(train_local.hour == 0) & (train_local.minute == 0)]
test_starts  = test_df.index[(test_local.hour == 0) & (test_local.minute == 0)]

train_blocks = eval_recursive_midnight_blocks(train_starts, "Train")
test_blocks  = eval_recursive_midnight_blocks(test_starts,  "Test")

# Plots
if len(train_blocks) > 0:
    plt.figure(figsize=(12, 4))
    plt.plot(train_blocks.index, train_blocks["MAE"].values)
    plt.title("Rekursiv: MAE_24h pro Tag (Train, Start 00:00 lokal)")
    plt.ylabel("MAE_24h")
    plt.tight_layout()
    plt.show()

if len(test_blocks) > 0:
    plt.figure(figsize=(12, 4))
    plt.plot(test_blocks.index, test_blocks["MAE"].values)
    plt.title("Rekursiv: MAE_24h pro Tag (Test, Start 00:00 lokal)")
    plt.ylabel("MAE_24h")
    plt.tight_layout()
    plt.show()

# Optional: Beispieltag Plot (Test)
if SHOW_EXAMPLE_DAY_PLOT and EXAMPLE_DAY_UTC in test_df.index:
    start_ts = EXAMPLE_DAY_UTC
    history_end = start_ts - pd.Timedelta("15min")
    history_start = history_end - pd.Timedelta(days=HISTORY_DAYS)
    history = df.loc[history_start:history_end].copy()

    fc = forecast_multistep(pipe, history, df_full=df, steps=HORIZON)
    truth = df.reindex(fc.index)[TARGET_COL]
    mask = truth.notna()

    y_true = truth[mask].values
    y_pred = fc.loc[truth.index[mask], "forecast"].values
    mae, rmse, r2, mape = metrics_1d(y_true, y_pred)

    idx_local = truth.index.tz_convert(TZ_LOCAL)

    plt.figure(figsize=(12, 4))
    plt.plot(idx_local[mask], y_true, label="True 24h")
    plt.plot(idx_local[mask], y_pred, label="Forecast 24h (rekursiv)")
    plt.title(f"Rekursiv 24h Beispieltag\nMAE={mae:.0f}, RMSE={rmse:.0f}, R2={r2:.3f}, MAPE={mape:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.show()
