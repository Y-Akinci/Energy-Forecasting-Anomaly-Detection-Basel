"""
PRODUCTION FORECASTING SYSTEM
==============================
Nutzt trainiertes V13 Model für 1h Forecasts
Kann auf neue Daten angewendet werden
"""

import os
import pandas as pd
import numpy as np
from joblib import load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Pfade
    MODEL_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V13_Final"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Forecasts"
    
    MODEL_FILE = os.path.join(MODEL_DIR, "production_model.joblib")
    FEATURES_FILE = os.path.join(MODEL_DIR, "production_features.npy")
    CONFIG_FILE = os.path.join(MODEL_DIR, "production_config.json")
    
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    # Forecast Settings
    FORECAST_START = "2024-01-01 00:00:00"  # Ab wann forecasten
    FORECAST_END = "2024-12-31 23:45:00"    # Bis wann forecasten
    
    PLOT_DPI = 150

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PRODUCTION FORECASTING SYSTEM")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n[1/4] Lade trainiertes Modell...")

# Load model
if not os.path.exists(Config.MODEL_FILE):
    raise FileNotFoundError(f"Model nicht gefunden: {Config.MODEL_FILE}")

model = load(Config.MODEL_FILE)
print(f"Modell geladen: {Config.MODEL_FILE}")

# Load features
features = np.load(Config.FEATURES_FILE, allow_pickle=True)
print(f"Features ({len(features)}): {list(features[:5])}...")

# Load config
with open(Config.CONFIG_FILE, 'r') as f:
    config = json.load(f)
print(f"Model: {config['model_name']}")
print(f"Training RMSE: {config['test_rmse']:.2f}")
print(f"Training R²: {config['test_r2']:.4f}")

# ============================================================
# LOAD DATA
# ============================================================

print("\n[2/4] Lade Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

df_forecast = df.loc[Config.FORECAST_START:Config.FORECAST_END].copy()
print(f"Forecast Zeitraum: {df_forecast.index.min()} -> {df_forecast.index.max()}")
print(f"Zeilen: {len(df_forecast)}")

# ============================================================
# PREPARE FEATURES
# ============================================================

print("\n[3/4] Bereite Features vor...")

# Target
y_true = df_forecast['Stromverbrauch'].copy()

# Features
X = df_forecast.drop(columns=['Stromverbrauch']).copy()

# Object-Spalten entfernen
obj_cols = X.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    X = X.drop(columns=obj_cols)

X = X.astype(float)

# Nur Modell-Features
X = X[features]

# NaNs check
n_missing = X.isnull().sum().sum()
if n_missing > 0:
    print(f"WARNUNG: {n_missing} fehlende Werte werden mit Median gefüllt")
    X = X.fillna(X.median())

print(f"Features vorbereitet: {X.shape}")

# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\n[4/4] Erstelle Forecasts...")

y_pred = model.predict(X)

print(f"Forecasts erstellt: {len(y_pred)}")

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\nForecast Performance:")
print(f"  RMSE: {rmse:.2f} kWh")
print(f"  MAE:  {mae:.2f} kWh")
print(f"  R²:   {r2:.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================

# DataFrame mit Ergebnissen
results_df = pd.DataFrame({
    'datetime': df_forecast.index,
    'true': y_true.values,
    'predicted': y_pred,
    'residual': y_true.values - y_pred,
    'abs_error': np.abs(y_true.values - y_pred),
})

results_path = os.path.join(Config.OUTPUT_DIR, f'forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
results_df.to_csv(results_path, index=False)
print(f"\nErgebnisse gespeichert: {results_path}")

# ============================================================
# PLOTS
# ============================================================

print("\nErstelle Plots...")

# Plot 1: Full Year Forecast
fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(df_forecast.index, y_true, label='True', linewidth=1, alpha=0.7)
ax.plot(df_forecast.index, y_pred, label='Forecast', linewidth=1, linestyle='--', alpha=0.7)
ax.set_title('1h Forecast - Full Year 2024')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_full_year.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 2: Last Week Detail
fig, ax = plt.subplots(figsize=(16, 6))

end_time = df_forecast.index[-1]
start_time = end_time - pd.Timedelta(days=7)
mask = (df_forecast.index >= start_time) & (df_forecast.index <= end_time)

ax.plot(df_forecast.index[mask], y_true[mask], label='True', linewidth=1.5)
ax.plot(df_forecast.index[mask], y_pred[mask], label='Forecast', linewidth=1.5, linestyle='--')
ax.set_title('1h Forecast - Last 7 Days')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_last_week.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 3: Residuals Over Time
fig, ax = plt.subplots(figsize=(20, 6))

residuals = y_true.values - y_pred
ax.scatter(df_forecast.index, residuals, s=1, alpha=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(rmse, color='orange', linestyle=':', linewidth=1, label=f'±RMSE ({rmse:.0f})')
ax.axhline(-rmse, color='orange', linestyle=':', linewidth=1)
ax.set_title('Residuals Over Time')
ax.set_xlabel('Zeit')
ax.set_ylabel('Residual (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'residuals_timeline.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 4: Residuals Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
axes[0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residual (kWh)')
axes[0].set_ylabel('Count')
axes[0].set_title('Residuals Distribution')
axes[0].grid(alpha=0.3)

# QQ Plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'residuals_distribution.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 5: Error by Hour of Day
fig, ax = plt.subplots(figsize=(12, 6))

results_df['hour'] = pd.to_datetime(results_df['datetime']).dt.hour
hourly_mae = results_df.groupby('hour')['abs_error'].mean()

ax.bar(hourly_mae.index, hourly_mae.values, alpha=0.7)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean Absolute Error (kWh)')
ax.set_title('Forecast Error by Hour of Day')
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'error_by_hour.png'), dpi=Config.PLOT_DPI)
plt.close()

print(f"Plots gespeichert in: {Config.OUTPUT_DIR}")

# ============================================================
# SUMMARY
# ============================================================

summary = {
    'forecast_date': datetime.now().isoformat(),
    'forecast_period': {
        'start': Config.FORECAST_START,
        'end': Config.FORECAST_END,
        'n_samples': len(y_pred),
    },
    'model_info': {
        'model_name': config['model_name'],
        'training_rmse': config['test_rmse'],
        'training_r2': config['test_r2'],
    },
    'forecast_performance': {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
    },
    'residual_stats': {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
    },
}

summary_path = os.path.join(Config.OUTPUT_DIR, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary gespeichert: {summary_path}")

print("\n" + "=" * 70)
print("FORECASTING ABGESCHLOSSEN")
print("=" * 70)
print(f"\nErgebnisse:")
print(f"  CSV:     {results_path}")
print(f"  Plots:   {Config.OUTPUT_DIR}")
print(f"  Summary: {summary_path}")
