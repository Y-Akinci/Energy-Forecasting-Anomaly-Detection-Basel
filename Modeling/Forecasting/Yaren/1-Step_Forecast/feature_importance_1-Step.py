import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =======================================
# Pfade
# =======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_1step_pipeline.joblib")
INPUT_CSV = os.path.join(DATA_DIR, "processed_merged_features.csv")


# =======================================
# Modell laden
# =======================================

pipe = joblib.load(MODEL_PATH)
print("Modell geladen:", MODEL_PATH)


# =======================================
# Daten laden
# =======================================

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

TARGET_COL = "Stromverbrauch"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# =======================================
# 1) XGBoost interne Feature Importance
# =======================================

def plot_xgb_importance(pipe, top_n=30):
    model = pipe.named_steps["model"]
    booster = model.get_booster()

    pre = pipe.named_steps["preprocess"]
    feature_names = []

    for name, trans, cols in pre.transformers_:
        if trans == "passthrough":
            feature_names.extend(cols)
        else:
            feature_names.extend(trans.get_feature_names_out(cols))

    gain = booster.get_score(importance_type="gain")

    rows = []
    for i, fname in enumerate(feature_names):
        rows.append([fname, gain.get(f"f{i}", 0.0)])

    imp = (
        pd.DataFrame(rows, columns=["Feature", "Gain"])
        .sort_values("Gain", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 0.35 * top_n + 2))
    plt.barh(imp["Feature"][::-1], imp["Gain"][::-1])
    plt.title("XGBoost Feature Importance (1-Step, Gain)")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


    print(imp.to_string(index=False))


plot_xgb_importance(pipe)


# =======================================
# 2) Permutation Importance (1-Step)
# =======================================

def plot_permutation_importance(pipe, X, y, sample_size=8000):
    if len(X) > sample_size:
        Xs = X.sample(sample_size, random_state=42)
        ys = y.loc[Xs.index]
    else:
        Xs, ys = X, y

    r = permutation_importance(
        pipe,
        Xs,
        ys,
        n_repeats=3,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    imp = (
        pd.DataFrame({
            "Feature": Xs.columns,
            "Importance": r.importances_mean
        })
        .sort_values("Importance", ascending=False)
        .head(30)
    )

    plt.figure(figsize=(10, 8))
    plt.barh(imp["Feature"][::-1], imp["Importance"][::-1])
    plt.title("Permutation Importance (1-Step)")
    plt.tight_layout()
    plt.show()

    print(imp.to_string(index=False))


plot_permutation_importance(pipe, X, y)


# =======================================
# 3) Tages-Prediction (1-Step)
# =======================================

def plot_day_1step(pipe, X, y, day_utc):
    day_start = pd.Timestamp(f"{day_utc} 00:00:00", tz="UTC")
    day_end = day_start + pd.Timedelta("24h") - pd.Timedelta("15min")

    mask = (y.index >= day_start) & (y.index <= day_end)
    Xd = X.loc[mask]
    yt = y.loc[mask]
    yp = pipe.predict(Xd)

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)

    idx_local = yt.index.tz_convert("Europe/Zurich")

    plt.figure(figsize=(12, 4))
    plt.plot(idx_local, yt.values, label="True")
    plt.plot(idx_local, yp, label="Pred")
    plt.title(
        f"1-Step Forecast {day_utc}\n"
        f"MAE={mae:.0f}, RMSE={rmse:.0f}, R2={r2:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)


# Beispiel
plot_day_1step(pipe, X, y, "2024-11-15")
