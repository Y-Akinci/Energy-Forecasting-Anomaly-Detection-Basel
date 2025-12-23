import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Pfade
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
MODEL_PATH = os.path.join(DATA_DIR, "rf_model.joblib")
FEAT_PATH = os.path.join(DATA_DIR, "rf_features.npy")

print("Lade Daten aus:", INPUT_FILE)
df_full = pd.read_csv(
    INPUT_FILE,
    sep=";",
    encoding="utf-8",
    parse_dates=["DateTime"],
    index_col="DateTime",
).sort_index()

# 1) Trainingszeitraum wie beim Training
train_start = "2021-01-01 00:00:00"
train_end   = "2024-12-31 23:45:00"
df_train = df_full.loc[train_start:train_end].copy()

target_col = "Stromverbrauch"
df_train[target_col] = df_train[target_col].astype(float)
X_all = df_train.drop(columns=[target_col]).copy()

# Zeitfeatures ergänzen
X_all["hour"] = X_all.index.hour
X_all["dayofweek"] = X_all.index.dayofweek
X_all["month"] = X_all.index.month
X_all["dayofyear"] = X_all.index.dayofyear

# Textspalten entfernen
obj_cols = X_all.select_dtypes(include="object").columns.tolist()
X_all = X_all.drop(columns=obj_cols)
X_all = X_all.astype(float)

# Featureliste laden und Spalten anpassen
feature_names = np.load(FEAT_PATH, allow_pickle=True)
X_train = X_all.reindex(columns=feature_names, fill_value=0.0)

# Modell laden
print("Lade Modell aus:", MODEL_PATH)
model = load(MODEL_PATH)

# ============================================================
# 2) ONE-STEP-FORECAST (15 Minuten nach Trainingsende)
# ============================================================
last_time = df_train.index[-1]              # 2024-12-31 23:45
next_time = last_time + pd.Timedelta("15min")

print("\nONE-STEP (15min ahead)")
print("Letzter Trainingszeitpunkt:", last_time)
print("Vorhersagezeitpunkt      :", next_time)

df_for_one = df_full.loc[:last_time].copy()
df_for_one[target_col] = df_for_one[target_col].astype(float)

# Zeitfeatures für next_time
df_for_one.loc[next_time, "hour"] = next_time.hour
df_for_one.loc[next_time, "dayofweek"] = next_time.dayofweek
df_for_one.loc[next_time, "month"] = next_time.month
df_for_one.loc[next_time, "dayofyear"] = next_time.dayofyear

# Lags & Diff
df_for_one[target_col] = df_for_one[target_col].astype(float)
df_for_one["Lag_15min"] = df_for_one[target_col].shift(1)
df_for_one["Lag_30min"] = df_for_one[target_col].shift(2)
df_for_one["Lag_1h"]    = df_for_one[target_col].shift(4)
df_for_one["Lag_24h"]   = df_for_one[target_col].shift(96)
df_for_one["Diff_15min"] = df_for_one[target_col] - df_for_one["Lag_15min"]

X_next = df_for_one.drop(columns=[target_col], errors="ignore")
X_next = X_next.drop(columns=X_next.select_dtypes(include="object").columns, errors="ignore")
X_next = X_next.astype(float)
X_next = X_next.reindex(columns=feature_names, fill_value=0.0)

x_one = X_next.loc[[next_time]]
y_hat_one = model.predict(x_one)[0]

print("Vorhersage:", next_time, "=>", y_hat_one)

if next_time in df_full.index:
    y_true_one = float(df_full.loc[next_time, target_col])
    err_one = y_hat_one - y_true_one
    print(f"Echter Wert : {y_true_one:.2f}")
    print(f"Fehler      : {err_one:.2f}")
    print(f"|Fehler|    : {abs(err_one):.2f}")
else:
    print("Kein echter Messwert für", next_time,)

