import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "common").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "common").exists():
    raise RuntimeError(f"Projektroot mit 'common' nicht gefunden. Start war: {Path(__file__).resolve()}")

sys.path.insert(0, str(PROJECT_ROOT))  

# common imports
import common.data as data
import common.metrics as m

load_dataset_and_split = data.load_dataset_and_split
train_test_metrics = m.train_test_metrics

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


# Daten aus common.data

bundle = load_dataset_and_split(verbose=True)

ROOT = bundle["root"]
df = bundle["df_full"]
train_df = bundle["train_df"]
test_df = bundle["test_df"]
NUMERIC_FEATURES = bundle["numeric_features"]
TARGET_COL = bundle["cfg"].target_col

X_train = train_df[NUMERIC_FEATURES]
X_test  = test_df[NUMERIC_FEATURES]
y_train = train_df[TARGET_COL].astype("float64")
y_test  = test_df[TARGET_COL].astype("float64")


# Preprocessor (nur numerisch)
preprocessor = ColumnTransformer(
    transformers=[("num", "passthrough", NUMERIC_FEATURES)],
    remainder="drop"
)



# Modelle

MODELS = {
    #"xgb": XGBRegressor(
    #    n_estimators=300,
    #    max_depth=6,
    #    subsample=0.8,
    #    colsample_bytree=0.8,
    #    learning_rate=0.05,
    #    n_jobs=-1,
    #    random_state=42,
    #),
    #"rf": RandomForestRegressor(
    #    n_estimators=300,
    #    max_depth=15,
    #    n_jobs=-1,
    #    random_state=42,
    #),
    "lgbm": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}


# Train/Test pro Modell + speichern

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

results = {}
best_name = None
best_pipe = None
best_pred_test = None
best_score = None

for name, model in MODELS.items():
    print(f"\n=== Training {name.upper()} (1-Step) ===")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    metrics = train_test_metrics(y_train, y_pred_train, y_test, y_pred_test)
    results[name] = metrics

    print(f"MAE_test  : {metrics['MAE_test']:.3f}")
    print(f"RMSE_test : {metrics['RMSE_test']:.3f}")
    print(f"R2_test   : {metrics['R2_test']:.4f}")
    print(f"MAPE_test : {metrics['MAPE_%_test']:.2f} %")

    # speichern
    model_path = MODEL_DIR / f"{name}_1step_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print("Gespeichert:", model_path)

    # bestes Modell merken (RMSE_test minimal)
    score = metrics["RMSE_test"]
    if best_score is None or score < best_score:
        best_score = score
        best_name = name
        best_pipe = pipe
        best_pred_test = y_pred_test


# Vergleichstabelle ausgeben

results_df = pd.DataFrame(results).T
print("\n=== MODELLVERGLEICH (1-Step) ===")
print(results_df[[
    "MAE_train", "RMSE_train", "R2_train", "MAPE_%_train",
    "MAE_test", "RMSE_test", "R2_test", "MAPE_%_test",
]])


print(f"\nBESTES MODELL (nach RMSE_test): {best_name.upper()} | RMSE_test={best_score:.3f}")


# 1-Step Plot: 15.11.2024 (24h Ausschnitt)
y_pred_test_series = pd.Series(best_pred_test, index=y_test.index)

target_start = pd.Timestamp("2024-11-15 00:00:00", tz="UTC")
target_end = target_start + pd.Timedelta("24h") - pd.Timedelta("15min")

mask = (y_test.index >= target_start) & (y_test.index <= target_end)

y_true_15 = y_test.loc[mask]
y_pred_15 = y_pred_test_series.loc[mask]

print("\n1-Step 15.11. | Punkte:", len(y_true_15))
print("Zeitraum:", y_true_15.index.min(), "→", y_true_15.index.max())

day_m = m.regression_metrics(y_true_15.values, y_pred_15.values)
mae_15 = day_m["MAE"]
rmse_15 = day_m["RMSE"]
r2_15 = day_m["R2"]

idx_local = y_true_15.index.tz_convert("Europe/Zurich")

plt.figure(figsize=(12, 4))
plt.plot(idx_local, y_true_15.values, label="True 1-Step (15.11.)")
plt.plot(idx_local, y_pred_15.values, label=f"Pred 1-Step ({best_name.upper()})")
plt.title(f"1-Step Forecast fuer 15.11.2024 ({best_name.upper()})\nMAE={mae_15:.0f}, RMSE={rmse_15:.0f}, R2={r2_15:.3f}")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n=== 1-Step Forecast 15.11.2024 ({best_name.upper()}) ===")
print(f"MAE  : {mae_15:.2f}")
print(f"RMSE : {rmse_15:.2f}")
print(f"R2   : {r2_15:.4f}")


# bestes Modell extra speichern (klarer Name)
best_path = MODEL_DIR / "best_1step_pipeline.joblib"
joblib.dump(best_pipe, best_path)
print("\nBestes Modell gespeichert unter:", best_path)
