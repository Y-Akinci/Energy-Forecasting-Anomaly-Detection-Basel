import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "common").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "common").exists():
    raise RuntimeError(f"Projektroot mit 'common' nicht gefunden. Start war: {Path(__file__).resolve()}")

sys.path.insert(0, str(PROJECT_ROOT))  # insert(0) ist robuster als append


# common imports
import common.data as data

load_dataset_and_split = data.load_dataset_and_split


# SETTINGS
MODEL_FILENAME = "best_1step_pipeline.joblib"      # wird aus /models geladen
N_REPEATS = 10                                     
SAMPLE_SIZE = 20000                                # permute auf Sample
RANDOM_STATE = 42
TOP_N = 25

SAVE_CSV = True
SAVE_PNG = True

# Daten laden
bundle = load_dataset_and_split(verbose=True)

ROOT_DIR = bundle["root"]
train_df = bundle["train_df"]
test_df = bundle["test_df"]
NUMERIC_FEATURES = bundle["numeric_features"]
TARGET_COL = bundle["cfg"].target_col

X_test = test_df[NUMERIC_FEATURES]
y_test = test_df[TARGET_COL].astype("float64")

# Modell laden
MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model nicht gefunden: {MODEL_PATH}")

pipe = joblib.load(MODEL_PATH)
print("Geladen:", MODEL_PATH)
try:
    pipe.set_output(transform="pandas")
    print("Pipeline Output: pandas (Feature-Namen bleiben erhalten)")
except Exception as e:
    print("set_output nicht verfuegbar (aeltere sklearn Version):", e)


# Subsample 
n = len(X_test)
if SAMPLE_SIZE is not None and SAMPLE_SIZE < n:
    # für Zeitreihen: nicht random mischen, sondern die letzten SAMPLE_SIZE nehmen
    X_eval = X_test.iloc[-SAMPLE_SIZE:].copy()
    y_eval = y_test.iloc[-SAMPLE_SIZE:].copy()
else:
    X_eval = X_test.copy()
    y_eval = y_test.copy()

print("Eval Samples:", len(X_eval))

# Permutation Importance
perm = permutation_importance(
    pipe,
    X_eval,
    y_eval,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

importances_mean = perm.importances_mean
importances_std = perm.importances_std

fi = pd.DataFrame({
    "feature": X_eval.columns,
    "importance_mean": importances_mean,
    "importance_std": importances_std,
}).sort_values("importance_mean", ascending=False)

print("\n=== TOP Feature Importance (Permutation, RMSE) ===")
print(fi.head(TOP_N).to_string(index=False))

# Speichern
out_dir = ROOT_DIR / "reports" / "feature_importance"
out_dir.mkdir(parents=True, exist_ok=True)

if SAVE_CSV:
    out_csv = out_dir / "feature_importance_1step_permutation.csv"
    fi.to_csv(out_csv, index=False)
    print("\nCSV gespeichert:", out_csv)

# Plot
top = fi.head(TOP_N).iloc[::-1]  # fuer horizontalen Plot: oben = groesster

plt.figure(figsize=(10, max(5, int(TOP_N * 0.35))))
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
plt.title(f"Permutation Feature Importance (1-Step) | Top {TOP_N}\nscoring=neg_RMSE, n_repeats={N_REPEATS}, sample={len(X_eval)}")
plt.xlabel("Importance (Delta neg_RMSE)")
plt.tight_layout()

if SAVE_PNG:
    out_png = out_dir / "feature_importance_1step_permutation.png"
    plt.savefig(out_png, dpi=200)
    print("PNG gespeichert:", out_png)

plt.show()
