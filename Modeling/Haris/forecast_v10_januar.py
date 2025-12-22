"""
V10 FORECAST - 7 TAGE PROGNOSE JANUAR 2025
==========================================
Rolling 1h Forecast für erste Woche Januar
Mit Validation gegen echte Daten
Zeigt Degradation über Zeit
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class Config:
    BASE_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V10_Production"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    FORECAST_DIR = os.path.join(BASE_DIR, "forecasts")
    PLOTS_DIR = os.path.join(BASE_DIR, "forecast_plots")
    
    for dir in [FORECAST_DIR, PLOTS_DIR]:
        os.makedirs(dir, exist_ok=True)
    
    MODEL_FILE = os.path.join(MODEL_DIR, "v10_model.joblib")
    FEATURES_FILE = os.path.join(MODEL_DIR, "v10_features.npy")
    CONFIG_FILE = os.path.join(MODEL_DIR, "v10_config.json")
    
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    VALIDATION_FILE = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\das hier anfang januar.csv"
    
    FORECAST_START = "2025-01-01 00:00:00"
    FORECAST_END = "2025-01-07 23:45:00"  # Nur 7 Tage
    
    PLOT_DPI = 150

print("=" * 80)
print("V10 FORECAST - 7 TAGE PROGNOSE JANUAR 2025")
print("=" * 80)

print("\n[1/5] Lade Model...")

model = load(Config.MODEL_FILE)
features = np.load(Config.FEATURES_FILE, allow_pickle=True)

with open(Config.CONFIG_FILE, 'r') as f:
    config = json.load(f)

print(f"Model: {config['model_name']}")
print(f"Features: {len(features)}")
print(f"Training R²: {config['test_r2']:.4f}")

print("\n[2/5] Lade historische Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

print(f"Historische Daten bis: {df.index[-1]}")

df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
df['weekday_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

print("\n[3/5] Bereite Forecast vor...")

forecast_start = pd.to_datetime(Config.FORECAST_START)
forecast_end = pd.to_datetime(Config.FORECAST_END)

forecast_dates = pd.date_range(
    start=forecast_start,
    end=forecast_end,
    freq='15min'
)

print(f"Forecast: {forecast_dates[0]} bis {forecast_dates[-1]}")
print(f"Steps: {len(forecast_dates):,}")

future_df = pd.DataFrame(index=forecast_dates)

future_df['hour_sin'] = np.sin(2 * np.pi * future_df.index.hour / 24)
future_df['hour_cos'] = np.cos(2 * np.pi * future_df.index.hour / 24)
future_df['month_sin'] = np.sin(2 * np.pi * (future_df.index.month - 1) / 12)
future_df['month_cos'] = np.cos(2 * np.pi * (future_df.index.month - 1) / 12)
future_df['weekday_sin'] = np.sin(2 * np.pi * future_df.index.dayofweek / 7)
future_df['weekday_cos'] = np.cos(2 * np.pi * future_df.index.dayofweek / 7)
future_df['IstSonntag'] = (future_df.index.dayofweek == 6).astype(int)
future_df['IstArbeitstag'] = (~future_df.index.dayofweek.isin([5, 6])).astype(int)

weather_features = [f for f in features if '_lag15' in f and f not in future_df.columns]

for wf in weather_features:
    if wf in df.columns:
        seasonal_hourly = df[wf].tail(30 * 96).groupby(df.tail(30 * 96).index.hour).mean()
        future_df[wf] = future_df.index.hour.map(seasonal_hourly)
        
        if future_df[wf].isnull().any():
            future_df[wf].fillna(df[wf].mean(), inplace=True)

print(f"Wetter-Features: {len(weather_features)} (seasonal averages)")

print("\n[4/5] Rolling Forecast...")

combined_df = pd.concat([df[['Stromverbrauch']], future_df])

predictions = []
prediction_times = []

total_steps = len(forecast_dates)

for i, timestamp in enumerate(forecast_dates):
    if i % 960 == 0 and i > 0:
        days_done = i // 96
        total_days = total_steps // 96
        pct = (i / total_steps) * 100
        print(f"  Tag {days_done}/{total_days} ({pct:.1f}%)")
    
    row = future_df.loc[timestamp].copy()
    
    lag_24h_time = timestamp - pd.Timedelta(hours=24)
    
    if lag_24h_time in combined_df.index:
        if lag_24h_time in future_df.index:
            try:
                idx = list(forecast_dates).index(lag_24h_time)
                row['Lag_24h'] = predictions[idx]
            except:
                row['Lag_24h'] = combined_df.loc[lag_24h_time, 'Stromverbrauch']
        else:
            row['Lag_24h'] = combined_df.loc[lag_24h_time, 'Stromverbrauch']
    else:
        hour = timestamp.hour
        row['Lag_24h'] = df['Stromverbrauch'].tail(30*96).groupby(df.tail(30*96).index.hour).mean()[hour]
    
    X_row = []
    for feat in features:
        if feat in row.index:
            X_row.append(row[feat])
        elif feat in df.columns:
            X_row.append(df[feat].mean())
        else:
            X_row.append(0)
    
    X_row = np.array(X_row).reshape(1, -1)
    
    pred = model.predict(X_row)[0]
    
    predictions.append(pred)
    prediction_times.append(timestamp)

print(f"\nForecast abgeschlossen: {len(predictions):,} Predictions")

print("\n[5/5] Validation & Aggregation...")

results_df = pd.DataFrame({
    'datetime': prediction_times,
    'predicted_15min': predictions,
})
results_df = results_df.set_index('datetime')

try:
    val_df = pd.read_csv(Config.VALIDATION_FILE, sep=";", encoding="utf-8")
    val_df['timestamp'] = pd.to_datetime(val_df['Start der Messung'])
    
    if val_df['timestamp'].dt.tz is not None:
        val_df['timestamp'] = val_df['timestamp'].dt.tz_localize(None)
    
    val_df = val_df.set_index('timestamp').sort_index()
    val_df = val_df.rename(columns={'Stromverbrauch': 'actual_15min'})
    
    print(f"Validation Daten: {val_df.index.min()} bis {val_df.index.max()}")
    
    # Direct merge on floored timestamps
    results_temp = results_df.copy()
    results_temp['timestamp_key'] = results_temp.index.floor('15min')
    val_temp = val_df.copy()
    val_temp['timestamp_key'] = val_temp.index.floor('15min')
    
    merged = results_temp.reset_index().merge(
        val_temp[['timestamp_key', 'actual_15min']],
        on='timestamp_key',
        how='left'
    )
    
    results_df['actual_15min'] = merged['actual_15min'].values
    
    valid_mask = results_df['actual_15min'].notna()
    n_valid = valid_mask.sum()
    
    print(f"Overlapping: {n_valid:,} / {len(results_df):,}")
    
    validation_available = True
    
    if n_valid > 0:
        valid_results = results_df[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(valid_results['actual_15min'], valid_results['predicted_15min']))
        mae = mean_absolute_error(valid_results['actual_15min'], valid_results['predicted_15min'])
        
        if n_valid > 1:
            r2 = r2_score(valid_results['actual_15min'], valid_results['predicted_15min'])
        else:
            r2 = np.nan
        
        mape = np.mean(np.abs((valid_results['actual_15min'] - valid_results['predicted_15min']) / valid_results['actual_15min'])) * 100
        
        print(f"\nValidation 15-min:")
        print(f"  RMSE: {rmse:.2f} kWh")
        print(f"  MAE:  {mae:.2f} kWh")
        if not np.isnan(r2):
            print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        results_df['error'] = results_df['actual_15min'] - results_df['predicted_15min']
    else:
        rmse = mae = r2 = mape = None
        print("Keine Validation möglich")
except Exception as e:
    print(f"Validation Fehler: {e}")
    validation_available = False
    rmse = mae = r2 = mape = None

# Hourly aggregation
results_df['hour'] = results_df.index.floor('H')

agg_dict_hourly = {'predicted_15min': 'sum'}
if validation_available and 'actual_15min' in results_df.columns:
    agg_dict_hourly['actual_15min'] = 'sum'

hourly = results_df.groupby('hour').agg(agg_dict_hourly).rename(
    columns={'predicted_15min': 'predicted_hourly', 'actual_15min': 'actual_hourly'}
)

# Daily aggregation
results_df['date'] = results_df.index.date

agg_dict_daily = {'predicted_15min': 'sum'}
if validation_available and 'actual_15min' in results_df.columns:
    agg_dict_daily['actual_15min'] = 'sum'

daily = results_df.groupby('date').agg(agg_dict_daily).rename(
    columns={'predicted_15min': 'predicted_daily', 'actual_15min': 'actual_daily'}
)

if 'actual_hourly' in hourly.columns and hourly['actual_hourly'].notna().sum() > 0:
    valid_hourly = hourly[hourly['actual_hourly'].notna()]
    hourly_rmse = np.sqrt(mean_squared_error(valid_hourly['actual_hourly'], valid_hourly['predicted_hourly']))
    print(f"\nValidation Stündlich: RMSE={hourly_rmse:.2f} kWh")

if 'actual_daily' in daily.columns and daily['actual_daily'].notna().sum() > 0:
    valid_daily = daily[daily['actual_daily'].notna()]
    daily_rmse = np.sqrt(mean_squared_error(valid_daily['actual_daily'], valid_daily['predicted_daily']))
    print(f"Validation Täglich: RMSE={daily_rmse:.2f} kWh")

forecast_path = os.path.join(Config.FORECAST_DIR, 'forecast_7tage_januar_15min.csv')
results_df.to_csv(forecast_path)

hourly_path = os.path.join(Config.FORECAST_DIR, 'forecast_7tage_januar_hourly.csv')
hourly.to_csv(hourly_path)

daily_path = os.path.join(Config.FORECAST_DIR, 'forecast_7tage_januar_daily.csv')
daily.to_csv(daily_path)

print(f"\nCSVs gespeichert:")
print(f"  15-min: {forecast_path}")
print(f"  Stündlich: {hourly_path}")
print(f"  Täglich: {daily_path}")

print("\nErstelle Plots...")

fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(results_df.index, results_df['predicted_15min'], label='Forecast', linewidth=0.8, alpha=0.8, color='blue')

if 'actual_15min' in results_df.columns:
    valid = results_df['actual_15min'].notna()
    if valid.sum() > 0:
        ax.plot(results_df.index[valid], results_df.loc[valid, 'actual_15min'],
                label='Actual', linewidth=0.8, alpha=0.8, color='green')

ax.set_title('15-min Forecast Erste 7 Tage Januar 2025')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'forecast_15min_full.png'), dpi=Config.PLOT_DPI)
plt.close()

fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(hourly.index, hourly['predicted_hourly'], label='Forecast', linewidth=1.5, alpha=0.8, color='blue', marker='o', markersize=2)

if 'actual_hourly' in hourly.columns:
    valid = hourly['actual_hourly'].notna()
    if valid.sum() > 0:
        ax.plot(hourly.index[valid], hourly.loc[valid, 'actual_hourly'],
                label='Actual', linewidth=1.5, alpha=0.8, color='green', marker='s', markersize=2)

ax.set_title('Stündliche Prognose Erste 7 Tage Januar 2025')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stündlicher Verbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'forecast_hourly.png'), dpi=Config.PLOT_DPI)
plt.close()

fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(daily.index, daily['predicted_daily'], label='Forecast', linewidth=2, alpha=0.8, color='blue', marker='o', markersize=6)

if 'actual_daily' in daily.columns:
    valid = daily['actual_daily'].notna()
    if valid.sum() > 0:
        ax.plot([d for d, v in zip(daily.index, valid) if v],
                daily.loc[valid, 'actual_daily'],
                label='Actual', linewidth=2, alpha=0.8, color='green', marker='s', markersize=6)

ax.set_title('Tägliche Prognose Erste 7 Tage Januar 2025')
ax.set_xlabel('Datum')
ax.set_ylabel('Tagesverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'forecast_daily.png'), dpi=Config.PLOT_DPI)
plt.close()

if 'error' in results_df.columns and results_df['error'].notna().sum() > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    valid = results_df['error'].notna()
    
    axes[0, 0].plot(results_df.index[valid], results_df.loc[valid, 'error'], linewidth=0.5, alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title('Error Over Time (15-min)')
    axes[0, 0].set_xlabel('Zeit')
    axes[0, 0].set_ylabel('Error (kWh)')
    axes[0, 0].grid(alpha=0.3)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    axes[0, 1].scatter(results_df.loc[valid, 'actual_15min'],
                      results_df.loc[valid, 'predicted_15min'], s=1, alpha=0.3)
    min_val = min(results_df.loc[valid, 'actual_15min'].min(),
                  results_df.loc[valid, 'predicted_15min'].min())
    max_val = max(results_df.loc[valid, 'actual_15min'].max(),
                  results_df.loc[valid, 'predicted_15min'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual (kWh)')
    axes[0, 1].set_ylabel('Predicted (kWh)')
    axes[0, 1].set_title('Predicted vs Actual')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].hist(results_df.loc[valid, 'error'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Error (kWh)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    hourly_error = results_df.loc[valid].groupby(results_df.loc[valid].index.hour)['error'].apply(lambda x: x.abs().mean())
    axes[1, 1].bar(hourly_error.index, hourly_error.values, alpha=0.7)
    axes[1, 1].set_xlabel('Stunde')
    axes[1, 1].set_ylabel('Mean Absolute Error (kWh)')
    axes[1, 1].set_title('Error by Hour of Day')
    axes[1, 1].set_xticks(range(0, 24, 2))
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'error_analysis.png'), dpi=Config.PLOT_DPI)
    plt.close()

print(f"Plots: {Config.PLOTS_DIR}")

summary = {
    'created': datetime.now().isoformat(),
    'forecast_period': {
        'start': str(forecast_dates[0]),
        'end': str(forecast_dates[-1]),
        'n_15min': len(predictions),
        'n_hours': len(hourly),
        'n_days': len(daily),
    },
    'model_info': {
        'model_name': config['model_name'],
        'training_r2': config['test_r2'],
    },
    'forecast_stats': {
        '15min_mean': float(np.mean(predictions)),
        '15min_std': float(np.std(predictions)),
        'hourly_mean': float(hourly['predicted_hourly'].mean()),
        'daily_mean': float(daily['predicted_daily'].mean()),
        'total_month': float(daily['predicted_daily'].sum()),
    },
}

if rmse is not None:
    summary['validation_15min'] = {
        'n_samples': int(n_valid),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2) if not np.isnan(r2) else None,
        'mape': float(mape),
    }

summary_path = os.path.join(Config.FORECAST_DIR, 'summary_7tage_januar.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary: {summary_path}")

print("\n" + "=" * 80)
print("FORECAST COMPLETE")
print("=" * 80)

if rmse is not None:
    print(f"\nValidation Performance:")
    print(f"  15-min RMSE: {rmse:.2f} kWh")
    print(f"  15-min MAPE: {mape:.2f}%")

print(f"\nGesamt 7 Tage Prognose: {daily['predicted_daily'].sum():.0f} kWh")