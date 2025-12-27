import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance


# SETTINGS
MODEL_FILENAME = "best_multioutput_24h_lgbm.joblib"  
CSV_FILENAME   = "processed_merged_features.csv"

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS   = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

TARGET_COL = "Stromverbrauch"
HORIZON = 96

TEST_SPLIT = 0.30   

# Plot
N_PLOT = 5000                 # für t+1 Zeitreihe
PERM_SAMPLE_SIZE = 8000
PERM_TOP_N = 25

# Logs/Warnungen reduzieren
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Pfade
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR
while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / MODEL_FILENAME
INPUT_CSV  = DATA_DIR / CSV_FILENAME

print("MODEL_PATH:", MODEL_PATH)
print("INPUT_CSV :", INPUT_CSV)

pipe = joblib.load(MODEL_PATH)
print("Modell geladen.")


# Daten laden
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

df = df.loc[START_TS:END_TS].copy()
df_truth = df.copy()

print("Zeitraum:", df.index.min(), "→", df.index.max(), "| Zeilen:", len(df))

# Multi-Output Targets bauen (t+1..t+96)
for h in range(1, HORIZON + 1):
    df[f"{TARGET_COL}_t+{h}"] = df[TARGET_COL].shift(-h)

TARGET_COLS = [f"{TARGET_COL}_t+{h}" for h in range(1, HORIZON + 1)]
df = df.dropna(subset=TARGET_COLS)
print("Nach Target-Shift:", df.index.min(), "→", df.index.max(), "| Zeilen:", len(df))


# Features sin/cos erstellen aus kat. Features

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

# Wetterspalten
WEATHER_ALL = [c for c in df.columns if c.endswith("_lag15") and c not in LAG_FEATURES]
WEATHER_FEATURES = [c for c in WEATHER_ALL if c not in EXCLUDE_WEATHER]

# Finale Features
FEATURES = []
FEATURES += CALENDAR_FEATURES
FEATURES += LAG_FEATURES
FEATURES += WEATHER_FEATURES

FEATURES = [f for f in FEATURES if f in df.columns]
print("Aktive Features:", len(FEATURES))

# Anzeige-Spalten droppen
DROP_COLS = [
    "Datum (Lokal)", "Zeit (Lokal)",
    "Monat", "Wochentag", "Stunde (Lokal)",
    "Tag des Jahres", "Quartal"
]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

X_all = df[FEATURES].copy()
y_all = df[TARGET_COLS].astype("float64")


# Train/Test Split (zeitlich)

n = len(df)
split_idx = int(n * (1 - TEST_SPLIT))

X_train = X_all.iloc[:split_idx]
X_test  = X_all.iloc[split_idx:]
y_train = y_all.iloc[:split_idx]
y_test  = y_all.iloc[split_idx:]

print("Train:", X_train.index.min(), "→", X_train.index.max(), "|", len(X_train))
print("Test :", X_test.index.min(), "→", X_test.index.max(), "|", len(X_test))

# =========================================================
# Predict
# =========================================================
y_pred_test = pipe.predict(X_test)  # shape: (n_test, 96)
y_true_test = y_test.values         # shape: (n_test, 96)

# =========================================================
# 1) Global geflattet (dein klassisches MAE ueber alles)
# =========================================================
mae_flat = mean_absolute_error(y_true_test.ravel(), y_pred_test.ravel())
print("\nGLOBAL (geflattet ueber alle Zeilen und alle 96 Horizonte)")
print(f"MAE_flat_test: {mae_flat:.3f}")

# =========================================================
# 2) 24h-Block-MAE pro Zeile, dann Durchschnitt (DAS willst du)
# =========================================================
mae_block_per_row = np.mean(np.abs(y_true_test - y_pred_test), axis=1)   # (n_test,)
mae_block_mean = float(np.mean(mae_block_per_row))

print("\n24h-Block (jede Zeile = 1x 24h Forecast, MAE ueber 96 Werte)")
print(f"MAE_24h_mean_test: {mae_block_mean:.3f}")

# =========================================================
# 3) Horizon-wise MAE (1..96)
# =========================================================
mae_per_h = np.mean(np.abs(y_true_test - y_pred_test), axis=0)  # (96,)
print("\nHORIZON-WISE")
print(f"MAE_t+1 : {mae_per_h[0]:.3f}")
print(f"MAE_t+96: {mae_per_h[-1]:.3f}")

# =========================================================
# PLOT A: t+1 Zeitreihe (durchgehend)
# =========================================================
t1_true = pd.Series(y_true_test[:, 0], index=y_test.index)
t1_pred = pd.Series(y_pred_test[:, 0], index=y_test.index)

# optional: ausduennen fuer Plot
if len(t1_true) > N_PLOT:
    t1_true_p = t1_true.iloc[:N_PLOT]
    t1_pred_p = t1_pred.iloc[:N_PLOT]
else:
    t1_true_p, t1_pred_p = t1_true, t1_pred

plt.figure(figsize=(12, 4))
plt.plot(t1_true_p.index, t1_true_p.values, label="True t+1")
plt.plot(t1_pred_p.index, t1_pred_p.values, label="Pred t+1")
plt.title("t+1 ueber Testset (durchgehende Reihe)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# PLOT B: MAE pro Horizont 1..96
# =========================================================
plt.figure(figsize=(12, 4))
plt.plot(np.arange(1, HORIZON + 1), mae_per_h)
plt.title("MAE pro Horizont (t+1 .. t+96)")
plt.xlabel("Horizont (15min Schritte)")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

# =========================================================
# PLOT C: Tages-Blocks (1 Start pro Tag um 00:00 lokal) -> MAE_24h ueber Zeit
# =========================================================
idx_local = y_test.index.tz_convert("Europe/Zurich")
is_midnight = (idx_local.hour == 0) & (idx_local.minute == 0)

y_test_mid = y_test.loc[is_midnight]
if len(y_test_mid) > 0:
    pos = np.where(is_midnight)[0]  # Positionen im Test-Array

    # MAE pro Tagesblock
    mae_day = []
    day_index = []
    for p in pos:
        mae = float(np.mean(np.abs(y_true_test[p, :] - y_pred_test[p, :])))
        mae_day.append(mae)
        day_index.append(y_test.index[p])

    s = pd.Series(mae_day, index=pd.DatetimeIndex(day_index))
    plt.figure(figsize=(12, 4))
    plt.plot(s.index, s.values)
    plt.title("MAE_24h pro Tag (Start 00:00 lokal)")
    plt.ylabel("MAE_24h")
    plt.tight_layout()
    plt.show()

    print("\nTagesblock-Auswertung (Start 00:00 lokal)")
    print("Anzahl Tage:", len(s))
    print("MAE_24h mean:", float(s.mean()))
else:
    print("\nKein 00:00 lokal im Test-Index gefunden (komisch, aber möglich).")

# =========================================================
# Permutation Importance (Multi-Output, neg MAE)
# =========================================================
def multioutput_neg_mae(estimator, X_in, y_in):
    y_pred = estimator.predict(X_in)
    yt = y_in.values.ravel()
    yp = y_pred.ravel()
    mae = np.mean(np.abs(yt - yp))
    return -mae

# Sample nur aus Test, damit es „ehrlich“ bleibt
X_perm = X_test
y_perm = y_test
if len(X_perm) > PERM_SAMPLE_SIZE:
    X_perm = X_perm.sample(PERM_SAMPLE_SIZE, random_state=42)
    y_perm = y_perm.loc[X_perm.index]

print(f"\nPermutation Importance auf {len(X_perm)} Test-Zeilen...")

r = permutation_importance(
    pipe,
    X_perm,
    y_perm,
    n_repeats=3,
    random_state=42,
    scoring=multioutput_neg_mae
)

imp = (
    pd.DataFrame({"Feature": X_perm.columns, "Importance": r.importances_mean})
    .sort_values("Importance", ascending=False)
    .head(PERM_TOP_N)
)

print("\nTop Features (Permutation Importance):")
print(imp.to_string(index=False))

plt.figure(figsize=(10, 0.35 * len(imp) + 2))
plt.barh(imp["Feature"][::-1], imp["Importance"][::-1])
plt.title("Permutation Importance (Multi-Output 24h, neg MAE)")
plt.tight_layout()
plt.show()
