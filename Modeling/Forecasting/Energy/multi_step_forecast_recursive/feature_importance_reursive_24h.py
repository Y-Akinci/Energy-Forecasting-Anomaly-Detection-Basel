import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Settings
TARGET_COL = "Stromverbrauch"
HORIZON = 96
TZ_LOCAL = "Europe/Zurich"

HISTORY_DAYS = 7
MAX_MISSING_PER_BLOCK = 4

MODEL_FILENAME = "best_recursive_1step_pipe.joblib"
FEATURES_FILENAME = "best_recursive_1step_features.txt"  

N_BLOCKS = 40
N_REPEATS = 5
RANDOM_STATE = 42
TOP_N = 25

SAVE_CSV = True
SAVE_PNG = True


# Projekt-Root 
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR
while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# Modell laden
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model nicht gefunden: {MODEL_PATH}")

pipe = joblib.load(MODEL_PATH)
print("Geladen:", MODEL_PATH)

# Feature-Liste laden 
feat_path = MODEL_DIR / FEATURES_FILENAME
if feat_path.exists():
    NUMERIC_FEATURES = [l.strip() for l in feat_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("Features geladen:", feat_path, "| n=", len(NUMERIC_FEATURES))
else:
    try:
        NUMERIC_FEATURES = list(pipe.named_steps["preprocess"].feature_names_in_)
        print("Features aus Pipeline genommen | n=", len(NUMERIC_FEATURES))
    except Exception:
        raise RuntimeError(
            "Keine Feature-Liste gefunden.\n"
            "Speicher bitte im Training best_recursive_1step_features.txt (eine Feature-Spalte pro Zeile)."
        )

# Daten laden
INPUT_CSV = DATA_DIR / "processed_merged_features.csv"
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"CSV nicht gefunden: {INPUT_CSV}")

df = pd.read_csv(
    INPUT_CSV,
    sep=";",
    encoding="latin1",
    parse_dates=["Start der Messung (UTC)"]
)
df.set_index("Start der Messung (UTC)", inplace=True)
df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

# gleiche Zeitspanne wie bei dir
START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS   = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")
df = df.loc[START_TS:END_TS].copy()

# Fehlende Sin/Cos Features
def ensure_calendar_sincos(df_in: pd.DataFrame, wanted_features: list[str]) -> pd.DataFrame:
    df_out = df_in

    missing = [c for c in wanted_features if c not in df_out.columns]
    if not missing:
        return df_out

    # Monat_sin/cos
    if ("Monat_sin" in missing or "Monat_cos" in missing):
        if "Monat" not in df_out.columns:
            raise KeyError("Fehlt Basis-Spalte 'Monat' im CSV, kann Monat_sin/cos nicht bauen.")
        df_out["Monat_sin"] = np.sin(2 * np.pi * df_out["Monat"] / 12)
        df_out["Monat_cos"] = np.cos(2 * np.pi * df_out["Monat"] / 12)

    # Wochentag_sin/cos
    if ("Wochentag_sin" in missing or "Wochentag_cos" in missing):
        if "Wochentag" not in df_out.columns:
            raise KeyError("Fehlt Basis-Spalte 'Wochentag' im CSV, kann Wochentag_sin/cos nicht bauen.")
        df_out["Wochentag_sin"] = np.sin(2 * np.pi * df_out["Wochentag"] / 7)
        df_out["Wochentag_cos"] = np.cos(2 * np.pi * df_out["Wochentag"] / 7)

    # Stunde_sin/cos
    if ("Stunde_sin" in missing or "Stunde_cos" in missing):
        if "Stunde (Lokal)" not in df_out.columns:
            raise KeyError("Fehlt Basis-Spalte 'Stunde (Lokal)' im CSV, kann Stunde_sin/cos nicht bauen.")
        df_out["Stunde_sin"] = np.sin(2 * np.pi * df_out["Stunde (Lokal)"] / 24)
        df_out["Stunde_cos"] = np.cos(2 * np.pi * df_out["Stunde (Lokal)"] / 24)

    # TagJahr_sin/cos
    if ("TagJahr_sin" in missing or "TagJahr_cos" in missing):
        if "Tag des Jahres" not in df_out.columns:
            raise KeyError("Fehlt Basis-Spalte 'Tag des Jahres' im CSV, kann TagJahr_sin/cos nicht bauen.")
        df_out["TagJahr_sin"] = np.sin(2 * np.pi * df_out["Tag des Jahres"] / 365)
        df_out["TagJahr_cos"] = np.cos(2 * np.pi * df_out["Tag des Jahres"] / 365)

    still_missing = [c for c in wanted_features if c not in df_out.columns]
    if still_missing:
        raise KeyError(
            "Diese Features fehlen im df und konnten nicht rekonstruiert werden:\n"
            + "\n".join(still_missing)
        )

    return df_out

df = ensure_calendar_sincos(df, NUMERIC_FEATURES)

# Train/Test Split 
n = len(df)
split_idx = int(n * 0.7)
test_df = df.iloc[split_idx:].copy()

# 00:00 lokal Startpunkte
test_local = test_df.index.tz_convert(TZ_LOCAL)
candidate_starts = test_df.index[(test_local.hour == 0) & (test_local.minute == 0)]
if len(candidate_starts) == 0:
    raise RuntimeError("Keine 00:00-lokal Starts im Test gefunden.")

rng = np.random.default_rng(RANDOM_STATE)

candidate_starts = candidate_starts.sort_values()

if N_BLOCKS < len(candidate_starts):
    pick_pos = rng.choice(len(candidate_starts), size=N_BLOCKS, replace=False)
    block_starts = candidate_starts.take(pick_pos).sort_values()
else:
    block_starts = candidate_starts

# Rekursiv Score: mean RMSE ueber 24h-Blöcke
def rmse_1d(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def score_blocks(X_mod: pd.DataFrame) -> float:
    rmses = []

    for start_ts in block_starts:
        history_end = start_ts - pd.Timedelta("15min")
        history_start = history_end - pd.Timedelta(days=HISTORY_DAYS)
        hist_truth = df.loc[history_start:history_end].copy()
        if len(hist_truth) < 96:
            continue

        df_temp = hist_truth[[TARGET_COL]].copy().sort_index()
        preds = []

        for _ in range(HORIZON):
            last_ts = df_temp.index[-1]
            next_ts = last_ts + pd.Timedelta("15min")
            if next_ts not in X_mod.index:
                break

            row = X_mod.loc[next_ts, NUMERIC_FEATURES].copy()

            # Lags/Diff aus rekursivem Target überschreiben, wenn diese Features existieren
            if "Lag_15min" in row.index:
                row["Lag_15min"] = df_temp.iloc[-1][TARGET_COL]
            if "Lag_30min" in row.index and len(df_temp) >= 2:
                row["Lag_30min"] = df_temp.iloc[-2][TARGET_COL]
            if "Lag_1h" in row.index and len(df_temp) >= 4:
                row["Lag_1h"] = df_temp.iloc[-4][TARGET_COL]
            if "Lag_24h" in row.index and len(df_temp) >= 96:
                row["Lag_24h"] = df_temp.iloc[-96][TARGET_COL]
            if "Diff_15min" in row.index and len(df_temp) >= 2:
                row["Diff_15min"] = df_temp.iloc[-1][TARGET_COL] - df_temp.iloc[-2][TARGET_COL]

            X_row = pd.DataFrame([row.values], columns=NUMERIC_FEATURES, index=[next_ts])
            y_hat = pipe.predict(X_row)[0]
            preds.append((next_ts, y_hat))
            df_temp.loc[next_ts] = {TARGET_COL: y_hat}

        if len(preds) < 10:
            continue

        fc = pd.DataFrame(preds, columns=["ts", "forecast"]).set_index("ts")
        truth = df.reindex(fc.index)[TARGET_COL]

        if int(truth.isna().sum()) > MAX_MISSING_PER_BLOCK:
            continue

        mask = truth.notna()
        y_true = truth[mask].values
        y_pred = fc.loc[truth.index[mask], "forecast"].values
        if len(y_true) < 10:
            continue

        rmses.append(rmse_1d(y_true, y_pred))

    if len(rmses) == 0:
        return np.nan
    return float(np.mean(rmses))

# Base matrix + Base score
missing_for_X = [c for c in NUMERIC_FEATURES if c not in df.columns]
if missing_for_X:
    raise KeyError(
        "Folgende Features stehen in deiner Feature-Liste, fehlen aber im df:\n"
        + "\n".join(missing_for_X)
        + "\n\nFix: Feature-Liste muss zu deinem CSV/Feature-Engineering passen."
    )

X_base = df[NUMERIC_FEATURES].copy()
base = score_blocks(X_base)
print("\nBasis mean Block-RMSE:", f"{base:.3f}", "| Bloecke:", len(block_starts))

if not np.isfinite(base):
    raise RuntimeError(
        "Basis-Score ist NaN.\n"
        "Typische Gruende: zu viele Missing-Werte im Truth-Block oder History laeuft ins Leere.\n"
        "Test: setze N_BLOCKS=10 und MAX_MISSING_PER_BLOCK hoch oder pruefe Daten."
    )
# Permutation Importance
rows = []
test_idx = test_df.index

for feat in NUMERIC_FEATURES:
    deltas = []

    for _ in range(N_REPEATS):
        Xp = X_base.copy()
        vals = Xp.loc[test_idx, feat].values.copy()
        rng.shuffle(vals)
        Xp.loc[test_idx, feat] = vals

        s = score_blocks(Xp)
        if np.isfinite(s):
            deltas.append(s - base)

    mean_d = float(np.mean(deltas)) if deltas else np.nan
    std_d  = float(np.std(deltas)) if deltas else np.nan
    rows.append((feat, mean_d, std_d))
    print(f"{feat:<45} delta_RMSE_mean={mean_d:.4f} (repeats={len(deltas)})")

fi = pd.DataFrame(rows, columns=["feature", "delta_RMSE_mean", "delta_RMSE_std"]).sort_values("delta_RMSE_mean", ascending=False)

print("\n=== TOP Feature Importance (Rekursiv 24h) ===")
print(fi.head(TOP_N).to_string(index=False))

# Speichern + Plot
out_dir = ROOT / "reports" / "feature_importance"
out_dir.mkdir(parents=True, exist_ok=True)

if SAVE_CSV:
    out_csv = out_dir / "feature_importance_recursive_24h_permutation.csv"
    fi.to_csv(out_csv, index=False)
    print("\nCSV gespeichert:", out_csv)

top = fi.head(TOP_N).dropna().iloc[::-1]
plt.figure(figsize=(10, max(5, int(len(top) * 0.35))))
plt.barh(top["feature"], top["delta_RMSE_mean"], xerr=top["delta_RMSE_std"])
plt.title(
    f"Permutation Feature Importance (Rekursiv 24h) | Top {len(top)}\n"
    f"Score: mean 24h Block-RMSE, blocks={len(block_starts)}, repeats={N_REPEATS}"
)
plt.xlabel("Delta RMSE (hoeher = wichtiger)")
plt.tight_layout()

if SAVE_PNG:
    out_png = out_dir / "feature_importance_recursive_24h_permutation.png"
    plt.savefig(out_png, dpi=200)
    print("PNG gespeichert:", out_png)

plt.show()
