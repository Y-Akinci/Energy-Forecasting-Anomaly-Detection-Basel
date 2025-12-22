import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# =========================================================
# SETTINGS (nur hier anpassen)
# =========================================================

MODEL_FILENAME = "xgb_multioutput_24h_pipeline.joblib"   # falls anders: hier aendern
CSV_FILENAME = "processed_merged_features.csv"

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS   = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

TARGET_COL = "Stromverbrauch"
HORIZON = 96

SAMPLE_SIZE = 8000   # weniger = schneller
TOP_N = 30           # wie viele Features im Plot


# =========================================================
# Pfade bauen
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
INPUT_CSV = os.path.join(DATA_DIR, CSV_FILENAME)

print("MODEL_PATH:", MODEL_PATH)
print("INPUT_CSV :", INPUT_CSV)


# =========================================================
# Modell laden
# =========================================================

print("MODEL_PATH exists?:", os.path.isfile(MODEL_PATH))
print("MODEL_DIR exists?:", os.path.isdir(MODEL_DIR))

if not os.path.isfile(MODEL_PATH):
    # Suche im ganzen Projekt nach der Datei, falls sie nicht im /models liegt
    found = None
    for dirpath, _, filenames in os.walk(ROOT):
        if MODEL_FILENAME in filenames:
            found = os.path.join(dirpath, MODEL_FILENAME)
            break

    if found is None:
        raise FileNotFoundError(
            f"Nicht gefunden: {MODEL_FILENAME}\n"
            f"Erwartet: {MODEL_PATH}\n"
            f"Run modling_multi_output.py und kopiere den ABS:-Pfad von dort."
        )

    print("Gefunden unter:", found)
    MODEL_PATH = found

pipe = joblib.load(MODEL_PATH)
print("Modell geladen.")


# =========================================================
# Daten laden
# =========================================================

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
print("Zeitraum:", df.index.min(), "→", df.index.max(), "| Zeilen:", len(df))


# =========================================================
# Multi-Output Targets bauen: Stromverbrauch_t+1 ... t+96
# =========================================================

for h in range(1, HORIZON + 1):
    df[f"{TARGET_COL}_t+{h}"] = df[TARGET_COL].shift(-h)

TARGET_COLS = [f"{TARGET_COL}_t+{h}" for h in range(1, HORIZON + 1)]
df = df.dropna(subset=TARGET_COLS)

X = df.drop(columns=TARGET_COLS)
y = df[TARGET_COLS].astype("float64")

print("X shape:", X.shape, "| y shape:", y.shape)


# =========================================================
# Permutation Importance (Multi-Output): eigenes Scoring
# =========================================================

def multioutput_neg_mae(estimator, X_in, y_in):
    y_pred = estimator.predict(X_in)       # (n,96)
    yt = y_in.values.ravel()
    yp = y_pred.ravel()
    mae = np.mean(np.abs(yt - yp))
    return -mae  # neg, weil "groesser ist besser" bei permutation_importance


def run_permutation_importance(pipe, X, y, sample_size=8000, top_n=30):
    if len(X) > sample_size:
        Xs = X.sample(sample_size, random_state=42)
        ys = y.loc[Xs.index]
    else:
        Xs, ys = X, y

    print(f"Permutation Importance auf {len(Xs)} Zeilen...")

    r = permutation_importance(
        pipe,
        Xs,
        ys,
        n_repeats=3,
        random_state=42,
        scoring=multioutput_neg_mae
    )

    imp = (
        pd.DataFrame({
            "Feature": Xs.columns,
            "Importance": r.importances_mean
        })
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    print("\n=== Top Features (Permutation Importance, Multi-Output 24h) ===")
    print(imp.to_string(index=False))

    plt.figure(figsize=(10, 0.35 * top_n + 2))
    plt.barh(imp["Feature"][::-1], imp["Importance"][::-1])
    plt.title("Permutation Importance (Multi-Output 24h)")
    plt.show(block=False)
    plt.pause(3)
    plt.close()


    return imp


# =========================================================
# GO
# =========================================================

run_permutation_importance(pipe, X, y, sample_size=SAMPLE_SIZE, top_n=TOP_N)
