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

# 2) 1 Tag vorhersagen: 2025-01-01
forecast_start = pd.Timestamp("2025-01-01 00:00:00", tz=df_full.index.tz)
forecast_end   = pd.Timestamp("2025-01-01 23:45:00", tz=df_full.index.tz)
freq = pd.Timedelta("15min")
future_index = pd.date_range(forecast_start, forecast_end, freq=freq, tz=df_full.index.tz)

print("Forecast-Zeitraum:", future_index[0], "→", future_index[-1], "Punkte:", len(future_index))

# Kopie für Forecast mit Lags
df_for_forecast = df_full.copy()
df_for_forecast[target_col] = df_for_forecast[target_col].astype(float)
df_for_forecast = df_for_forecast.loc[:train_end].copy()

forecast_values = []

for t in future_index:
    # Zeitfeatures
    df_for_forecast.loc[t, "hour"] = t.hour
    df_for_forecast.loc[t, "dayofweek"] = t.dayofweek
    df_for_forecast.loc[t, "month"] = t.month
    df_for_forecast.loc[t, "dayofyear"] = t.dayofyear

    # Lags & Diff
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

    # Vorhersage für spätere Lags eintragen
    df_for_forecast.loc[t, target_col] = y_hat_t

forecast_series = pd.Series(forecast_values, index=future_index, name="forecast")

print("Erste Forecast-Werte:")
print(forecast_series.head())

# 3) 1h / 3h / 6h Forecast-Werte direkt aus der Serie
#   1h  = 4 * 15min, 3h = 12 * 15min, 6h = 24 * 15min
start_t = future_index[0]
t_1h = start_t + pd.Timedelta("1H")
t_3h = start_t + pd.Timedelta("3H")
t_6h = start_t + pd.Timedelta("6H")

val_1h = forecast_series.loc[t_1h]
val_3h = forecast_series.loc[t_3h]
val_6h = forecast_series.loc[t_6h]

print("\nForecast-Horizonte ab", start_t)
print(" +1h:", t_1h, "=>", val_1h)
print(" +3h:", t_3h, "=>", val_3h)
print(" +6h:", t_6h, "=>", val_6h)

# 4) Vergleich mit echten Werten + je ein Plot
real_day = df_full[target_col].loc["2025-01-01":"2025-01-01"].astype(float)

common_idx = forecast_series.index.intersection(real_day.index)
if len(common_idx) > 0:
    y_true = real_day.loc[common_idx]
    y_pred = forecast_series.loc[common_idx]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("\nBacktesting 2025-01-01 (ganzer Tag):")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")

    # Gesamt-Plot wie bisher
    plt.figure(figsize=(14, 4))
    plt.plot(y_true.index, y_true.values, label="Ist", color="blue", linewidth=1)
    plt.plot(y_pred.index, y_pred.values, label="Forecast", color="red", linewidth=1, alpha=0.7)
    plt.title("Stromverbrauch 2025-01-01: Ist vs. Forecast")
    plt.xlabel("Zeit")
    plt.ylabel("Stromverbrauch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(DATA_DIR, "forecast_2025-01-01_full.png")
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print("Plot gespeichert unter:", plot_path)

    # Kleine Ausschnitte für 1h, 3h, 6h
    for horizon, t_h in [("1h", t_1h), ("3h", t_3h), ("6h", t_6h)]:
        end_h = t_h
        idx_h = (y_true.index >= start_t) & (y_true.index <= end_h)
        y_true_h = y_true[idx_h]
        y_pred_h = y_pred[idx_h]

        plt.figure(figsize=(8, 3))
        plt.plot(y_true_h.index, y_true_h.values, label="Ist", color="blue", linewidth=1)
        plt.plot(y_pred_h.index, y_pred_h.values, label="Forecast", color="red", linewidth=1, alpha=0.7)
        plt.title(f"Stromverbrauch 2025-01-01: 0–{horizon}")
        plt.xlabel("Zeit")
        plt.ylabel("Stromverbrauch")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plot_path_h = os.path.join(DATA_DIR, f"forecast_2025-01-01_{horizon}.png")
        plt.savefig(plot_path_h, dpi=100)
        plt.close()
        print(f"Plot {horizon} gespeichert unter:", plot_path_h)
else:
    print("Keine Überschneidung zwischen Forecast und echten Werten für 2025-01-01 gefunden.")

