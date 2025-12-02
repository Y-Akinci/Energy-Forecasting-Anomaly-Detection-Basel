import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# 1) Trainingszeitraum wie beim Training: 2021-01-01 bis 2024-12-31
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

# in float casten
X_all = X_all.astype(float)

# Featureliste laden und Spalten anpassen
feature_names = np.load(FEAT_PATH, allow_pickle=True)
X_train = X_all.reindex(columns=feature_names, fill_value=0.0)

# Modell laden
print("Lade Modell aus:", MODEL_PATH)
model = load(MODEL_PATH)

# 2) Januar 2025 autoregressiv vorhersagen
forecast_start = pd.Timestamp("2025-01-01 00:00:00", tz=df_full.index.tz)
forecast_end   = pd.Timestamp("2025-01-31 23:45:00", tz=df_full.index.tz)
freq = pd.Timedelta("15min")
future_index = pd.date_range(forecast_start, forecast_end, freq=freq, tz=df_full.index.tz)

print("Forecast-Zeitraum:", future_index[0], "→", future_index[-1], "Punkte:", len(future_index))

# Kopie für Forecast mit Lags
df_for_forecast = df_full.copy()
df_for_forecast[target_col] = df_for_forecast[target_col].astype(float)
df_for_forecast = df_for_forecast.loc[:train_end].copy()

forecast_values = []

for t in future_index:
    # Zeitfeatures für t
    df_for_forecast.loc[t, "hour"] = t.hour
    df_for_forecast.loc[t, "dayofweek"] = t.dayofweek
    df_for_forecast.loc[t, "month"] = t.month
    df_for_forecast.loc[t, "dayofyear"] = t.dayofyear

    # Lags & Diff wie im Training
    df_for_forecast[target_col] = df_for_forecast[target_col].astype(float)
    df_for_forecast["Lag_15min"] = df_for_forecast[target_col].shift(1)
    df_for_forecast["Lag_30min"] = df_for_forecast[target_col].shift(2)
    df_for_forecast["Lag_1h"]    = df_for_forecast[target_col].shift(4)
    df_for_forecast["Lag_24h"]   = df_for_forecast[target_col].shift(96)
    df_for_forecast["Diff_15min"] = df_for_forecast[target_col] - df_for_forecast["Lag_15min"]

    X_t = df_for_forecast.drop(columns=[target_col], errors="ignore")
    X_t = X_t.drop(columns=X_t.select_dtypes(include="object").columns, errors="ignore")
    X_t = X_t.astype(float)
    X_t = X_t.reindex(columns=feature_names, fill_value=0.0)

    x_last = X_t.loc[[t]]
    y_hat_t = model.predict(x_last)[0]
    forecast_values.append(y_hat_t)

    # Vorhersage als neuer Wert für spätere Lags
    df_for_forecast.loc[t, target_col] = y_hat_t

forecast_series = pd.Series(forecast_values, index=future_index, name="forecast")

print("Erste Forecast-Werte:")
print(forecast_series.head())

# 3) Vergleich mit echten Werten (Backtesting), falls vorhanden
real_jan = df_full[target_col].loc["2025-01-01":"2025-01-31"].astype(float)

common_idx = forecast_series.index.intersection(real_jan.index)
if len(common_idx) > 0:
    y_true = real_jan.loc[common_idx]
    y_pred = forecast_series.loc[common_idx]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae
