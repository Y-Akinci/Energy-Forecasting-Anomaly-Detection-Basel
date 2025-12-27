import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Settings
MODEL_FILENAME = "best_multioutput_24h_lgbm.joblib"
CSV_FILENAME = "processed_merged_features.csv"

TARGET_COL = "Stromverbrauch"
HORIZON = 96

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS   = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

TEST_SPLIT = 0.30
PERM_SAMPLE_SIZE = 8000
PERM_TOP_N = 25

# Projektpfade bestimmen
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR
while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# Modell laden
pipe = joblib.load(MODEL_DIR / MODEL_FILENAME)
print("Modell geladen:", MODEL_FILENAME)

# Daten laden
df = pd.read_csv(
    DATA_DIR / CSV_FILENAME,
    sep=";",
    encoding="latin1",
    parse_dates=["Start der Messung (UTC)"]
)

df.set_index("Start der Messung (UTC)", inplace=True)
df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
df = df.loc[START_TS:END_TS].copy()

# Multi-Output Targets (t+1 bis t+96)
for h in range(1, HORIZON + 1):
    df[f"{TARGET_COL}_t+{h}"] = df[TARGET_COL].shift(-h)

TARGET_COLS = [f"{TARGET_COL}_t+{h}" for h in range(1, HORIZON + 1)]
df = df.dropna(subset=TARGET_COLS)

# Zyklische Kalenderfeatures
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

WEATHER_FEATURES = [
    c for c in df.columns
    if c.endswith("_lag15") and c not in LAG_FEATURES
]

FEATURES = [
    f for f in (CALENDAR_FEATURES + LAG_FEATURES + WEATHER_FEATURES)
    if f in df.columns
]

# Spalten entfernen, die nur für Darstellung oder Feature-Erzeugung dienten
DROP_COLS = [
    "Datum (Lokal)", "Zeit (Lokal)",
    "Monat", "Wochentag",
    "Stunde (Lokal)", "Tag des Jahres", "Quartal"
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

X = df[FEATURES]
y = df[TARGET_COLS]

# Zeitlich konsistenter Test-Split
n = len(df)
split_idx = int(n * (1 - TEST_SPLIT))
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# Stichprobe Permutation Importance (Rechenzeit)
if len(X_test) > PERM_SAMPLE_SIZE:
    X_test = X_test.sample(PERM_SAMPLE_SIZE, random_state=42)
    y_test = y_test.loc[X_test.index]

# Bewertungsfunktion: negativer MAE über alle Horizonte
def neg_mae_multioutput(estimator, X_in, y_in):
    y_pred = estimator.predict(X_in).ravel()
    y_true = y_in.values.ravel()
    return -np.mean(np.abs(y_true - y_pred))

print("Berechne Permutation Feature Importance...")

r = permutation_importance(
    pipe,
    X_test,
    y_test,
    n_repeats=3,
    random_state=42,
    scoring=neg_mae_multioutput
)

imp = (
    pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": r.importances_mean
    })
    .sort_values("Importance", ascending=False)
    .head(PERM_TOP_N)
)

print("\nTop Features:")
print(imp.to_string(index=False))

# Ergebnisse speichern und visualisieren
out_dir = ROOT / "reports" / "feature_importance"
out_dir.mkdir(parents=True, exist_ok=True)

imp.to_csv(out_dir / "fi_multioutput_24h.csv", index=False)

plt.figure(figsize=(10, 0.35 * len(imp) + 2))
plt.barh(imp["Feature"][::-1], imp["Importance"][::-1])
plt.title("Permutation Feature Importance – Multi-Output 24h")
plt.tight_layout()
plt.savefig(out_dir / "fi_multioutput_24h.png", dpi=200)
plt.close()

print("Feature Importance berechnet und gespeichert.")
