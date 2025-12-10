"""
FUTURE FORECASTING WITH VALIDATION
===================================
Forecast bis 6. Dezember 2024
Validation gegen echte Daten
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
    BASE_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\Final"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    FORECAST_DIR = os.path.join(BASE_DIR, "forecasts")
    PLOTS_DIR = os.path.join(BASE_DIR, "forecast_plots")
    
    for dir in [FORECAST_DIR, PLOTS_DIR]:
        os.makedirs(dir, exist_ok=True)
    
    MODEL_FILE = os.path.join(MODEL_DIR, "final_model.joblib")
    FEATURES_FILE = os.path.join(MODEL_DIR, "final_features.npy")
    CONFIG_FILE = os.path.join(MODEL_DIR, "final_config.json")
    ENCODER_FILE = os.path.join(MODEL_DIR, "onehot_encoder.joblib")
    
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    VALIDATION_FILE = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\zum forecaste vergleichen.csv"
    
    FORECAST_TARGET_DATE = "2025-12-06"
    
    PLOT_DPI = 150

print("=" * 80)
print("FUTURE FORECASTING WITH VALIDATION")
print("=" * 80)

print("\n[1/6] Lade Model...")

model = load(Config.MODEL_FILE)
features = np.load(Config.FEATURES_FILE, allow_pickle=True)

with open(Config.CONFIG_FILE, 'r') as f:
    config = json.load(f)

try:
    encoder = load(Config.ENCODER_FILE)
    has_encoder = True
except:
    encoder = None
    has_encoder = False

print(f"Model: {config['model_name']}")
print(f"Features: {len(features)}")
print(f"OneHotEncoder: {'Ja' if has_encoder else 'Nein'}")

print("\n[2/6] Lade historische Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

# Remove timezone if present
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

print(f"Historische Daten bis: {df.index[-1]}")

print("\n[3/6] Lade Validierungs-Daten...")

try:
    val_df = pd.read_csv(
        Config.VALIDATION_FILE,
        sep=";",
        encoding="utf-8"
    )
    
    val_df['timestamp'] = pd.to_datetime(val_df['Start der Messung'])
    
    # Remove timezone if present
    if val_df['timestamp'].dt.tz is not None:
        val_df['timestamp'] = val_df['timestamp'].dt.tz_localize(None)
    
    val_df = val_df.set_index('timestamp').sort_index()
    val_df = val_df.rename(columns={'Stromverbrauch': 'actual_consumption'})
    
    print(f"Validierung Zeitraum: {val_df.index.min()} bis {val_df.index.max()}")
    print(f"Validierung Samples: {len(val_df):,}")
    
    validation_available = True
except Exception as e:
    print(f"Fehler beim Laden: {e}")
    validation_available = False
    val_df = None

print("\n[4/6] Bereite Forecast vor...")

last_hist_date = df.index[-1]
target_date = pd.to_datetime(Config.FORECAST_TARGET_DATE).replace(hour=23, minute=45)

# Remove timezone from last_hist_date if present
if last_hist_date.tz is not None:
    last_hist_date = last_hist_date.tz_localize(None)

# Ensure target_date has no timezone
if hasattr(target_date, 'tz') and target_date.tz is not None:
    target_date = target_date.tz_localize(None)

forecast_dates = pd.date_range(
    start=last_hist_date + pd.Timedelta(minutes=15),
    end=target_date,
    freq='15min'
)

print(f"Forecast von {forecast_dates[0]} bis {forecast_dates[-1]}")
print(f"Anzahl Steps: {len(forecast_dates):,}")

future_df = pd.DataFrame(index=forecast_dates)

future_df['Monat'] = future_df.index.month
future_df['Wochentag'] = future_df.index.dayofweek
future_df['Quartal'] = future_df.index.quarter
future_df['Stunde (Lokal)'] = future_df.index.hour
future_df['IstSonntag'] = (future_df.index.dayofweek == 6).astype(int)
future_df['IstArbeitstag'] = (~future_df.index.dayofweek.isin([5, 6])).astype(int)

weather_features = [f for f in features if '_lag15' in f and f not in future_df.columns]

for wf in weather_features:
    if wf in df.columns:
        seasonal_hourly = df[wf].tail(30 * 96).groupby(df.tail(30 * 96).index.hour).mean()
        future_df[wf] = future_df.index.hour.map(seasonal_hourly)
        
        if future_df[wf].isnull().any():
            future_df[wf].fillna(df[wf].mean(), inplace=True)

print(f"Wetter Features: {len(weather_features)}")

print("\n[5/6] Erstelle Rolling Forecast...")

combined_df = pd.concat([df[['Stromverbrauch']], future_df])

predictions = []
prediction_times = []

for i, timestamp in enumerate(forecast_dates):
    if i % 96 == 0 and i > 0:
        days_done = i // 96
        total_days = len(forecast_dates) // 96
        print(f"  Tag {days_done}/{total_days} abgeschlossen")
    
    row = future_df.loc[timestamp].copy()
    
    lag_15min_time = timestamp - pd.Timedelta(minutes=15)
    lag_30min_time = timestamp - pd.Timedelta(minutes=30)
    lag_1h_time = timestamp - pd.Timedelta(hours=1)
    lag_24h_time = timestamp - pd.Timedelta(hours=24)
    
    def get_lag_value(lag_time, predictions_list, prediction_times_list):
        if lag_time in combined_df.index:
            if lag_time in future_df.index:
                try:
                    idx = prediction_times_list.index(lag_time)
                    return predictions_list[idx]
                except:
                    pass
            return combined_df.loc[lag_time, 'Stromverbrauch']
        return df['Stromverbrauch'].tail(30*96).mean()
    
    row['Lag_15min'] = get_lag_value(lag_15min_time, predictions, prediction_times)
    row['Lag_30min'] = get_lag_value(lag_30min_time, predictions, prediction_times)
    row['Lag_1h'] = get_lag_value(lag_1h_time, predictions, prediction_times)
    row['Lag_24h'] = get_lag_value(lag_24h_time, predictions, prediction_times)
    
    X_row_dict = {}
    for feat in config['numerical_features']:
        if feat in row.index:
            X_row_dict[feat] = row[feat]
        elif feat in df.columns:
            X_row_dict[feat] = df[feat].mean()
        else:
            X_row_dict[feat] = 0
    
    X_num = pd.DataFrame([X_row_dict])
    
    if has_encoder and config['categorical_features']:
        X_cat_dict = {}
        for feat in config['categorical_features']:
            if feat in row.index:
                X_cat_dict[feat] = row[feat]
            else:
                X_cat_dict[feat] = 0
        
        X_cat = pd.DataFrame([X_cat_dict])
        X_cat_encoded = encoder.transform(X_cat)
        
        cat_feature_names = encoder.get_feature_names_out(config['categorical_features'])
        X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names)
        
        X_row = pd.concat([X_num.reset_index(drop=True), X_cat_encoded_df.reset_index(drop=True)], axis=1)
    else:
        X_row = X_num
    
    X_row = X_row[features]
    
    pred = model.predict(X_row)[0]
    
    predictions.append(pred)
    prediction_times.append(timestamp)

print(f"\nForecast abgeschlossen: {len(predictions):,} Predictions")

print("\n[6/6] Validation & Results...")

results_df = pd.DataFrame({
    'datetime': prediction_times,
    'predicted': predictions,
})
results_df = results_df.set_index('datetime')

if validation_available and val_df is not None:
    print(f"\nDebug Info:")
    print(f"  Forecast Index Type: {type(results_df.index[0])}")
    print(f"  Forecast First: {results_df.index[0]}")
    print(f"  Forecast Last: {results_df.index[-1]}")
    print(f"  Validation Index Type: {type(val_df.index[0])}")
    print(f"  Validation First: {val_df.index[0]}")
    print(f"  Validation Last: {val_df.index[-1]}")
    
    # Try direct merge on rounded timestamps
    results_df['timestamp_key'] = results_df.index.floor('15min')
    val_df['timestamp_key'] = val_df.index.floor('15min')
    
    # Merge on rounded timestamp
    merged = results_df.reset_index().merge(
        val_df[['actual_consumption']].reset_index(),
        on='timestamp_key',
        how='left',
        suffixes=('', '_val')
    )
    
    results_df['actual_consumption'] = merged['actual_consumption'].values
    
    valid_mask = results_df['actual_consumption'].notna()
    n_valid = valid_mask.sum()
    
    print(f"\nOverlapping timestamps: {n_valid:,} / {len(results_df):,}")
    
    print(f"\nOverlapping timestamps: {n_valid:,}")
    
    if n_valid > 0:
        valid_results = results_df[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(valid_results['actual_consumption'], valid_results['predicted']))
        mae = mean_absolute_error(valid_results['actual_consumption'], valid_results['predicted'])
        
        if n_valid > 1:
            r2 = r2_score(valid_results['actual_consumption'], valid_results['predicted'])
        else:
            r2 = np.nan
        
        mape = np.mean(np.abs((valid_results['actual_consumption'] - valid_results['predicted']) / valid_results['actual_consumption'])) * 100
        
        print(f"\nValidation Results ({n_valid:,} samples):")
        print(f"  RMSE: {rmse:.2f} kWh")
        print(f"  MAE:  {mae:.2f} kWh")
        if not np.isnan(r2):
            print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        results_df['error'] = results_df['actual_consumption'] - results_df['predicted']
        results_df['abs_error'] = np.abs(results_df['error'])
        results_df['pct_error'] = (results_df['error'] / results_df['actual_consumption'] * 100)
        
        print(f"\nDurchschnittlicher Fehler: {results_df['error'].mean():.2f} kWh")
        print(f"Max Fehler: {results_df['abs_error'].max():.2f} kWh")
    else:
        print("\nKeine überlappenden Daten für Validation")
        print(f"Forecast Range: {results_df.index.min()} bis {results_df.index.max()}")
        print(f"Validation Range: {val_df.index.min()} bis {val_df.index.max()}")
        rmse = mae = r2 = mape = None
else:
    print("\nKeine Validation möglich")
    rmse = mae = r2 = mape = None

forecast_path = os.path.join(Config.FORECAST_DIR, f'forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
results_df.to_csv(forecast_path)
print(f"\nForecast CSV: {forecast_path}")

print("\nErstelle Plots...")

fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(results_df.index, results_df['predicted'], label='Forecast', linewidth=1.5, color='blue', alpha=0.8)

if validation_available and 'actual_consumption' in results_df.columns:
    valid = results_df['actual_consumption'].notna()
    if valid.sum() > 0:
        ax.plot(results_df.index[valid], results_df.loc[valid, 'actual_consumption'],
                label='Actual', linewidth=1.5, color='green', alpha=0.8)

ax.set_title('Forecast bis 6. Dezember 2024 mit Validation')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'forecast_full.png'), dpi=Config.PLOT_DPI)
plt.close()

if validation_available and 'error' in results_df.columns and results_df['error'].notna().sum() > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    valid = results_df['error'].notna()
    
    axes[0, 0].plot(results_df.index[valid], results_df.loc[valid, 'error'], linewidth=1, alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title('Prediction Error Over Time')
    axes[0, 0].set_xlabel('Zeit')
    axes[0, 0].set_ylabel('Error (kWh)')
    axes[0, 0].grid(alpha=0.3)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    axes[0, 1].scatter(results_df.loc[valid, 'actual_consumption'],
                      results_df.loc[valid, 'predicted'], s=1, alpha=0.5)
    
    min_val = min(results_df.loc[valid, 'actual_consumption'].min(),
                  results_df.loc[valid, 'predicted'].min())
    max_val = max(results_df.loc[valid, 'actual_consumption'].max(),
                  results_df.loc[valid, 'predicted'].max())
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
    
    hourly_error = results_df.loc[valid].groupby(results_df.loc[valid].index.hour)['abs_error'].mean()
    axes[1, 1].bar(hourly_error.index, hourly_error.values, alpha=0.7)
    axes[1, 1].set_xlabel('Stunde')
    axes[1, 1].set_ylabel('Durchschnittlicher Absoluter Fehler (kWh)')
    axes[1, 1].set_title('Error by Hour of Day')
    axes[1, 1].set_xticks(range(0, 24, 2))
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'validation_analysis.png'), dpi=Config.PLOT_DPI)
    plt.close()

fig, ax = plt.subplots(figsize=(14, 6))

hourly_avg = results_df.groupby(results_df.index.hour)['predicted'].mean()
ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8, color='blue')
ax.set_xlabel('Stunde')
ax.set_ylabel('Durchschnittlicher Verbrauch (kWh)')
ax.set_title('Durchschnittliches Tagesprofil (Forecast Periode)')
ax.set_xticks(range(0, 24))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'hourly_pattern.png'), dpi=Config.PLOT_DPI)
plt.close()

results_df['date'] = results_df.index.date
daily_consumption = results_df.groupby('date')['predicted'].sum()

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(daily_consumption)), daily_consumption.values, alpha=0.7, color='steelblue')
ax.set_xlabel('Tag')
ax.set_ylabel('Tagesverbrauch (kWh)')
ax.set_title('Prognostizierter Tagesverbrauch')
ax.set_xticks(range(len(daily_consumption)))
ax.set_xticklabels([str(d) for d in daily_consumption.index], rotation=45, ha='right')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'daily_consumption.png'), dpi=Config.PLOT_DPI)
plt.close()

print(f"Plots: {Config.PLOTS_DIR}")

summary = {
    'forecast_created': datetime.now().isoformat(),
    'forecast_period': {
        'start': str(prediction_times[0]),
        'end': str(prediction_times[-1]),
        'n_predictions': len(predictions),
    },
    'model_info': {
        'model_name': config['model_name'],
        'training_rmse': config['test_rmse'],
        'training_r2': config['test_r2'],
    },
    'forecast_stats': {
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
    },
}

if validation_available and rmse is not None:
    summary['validation'] = {
        'n_samples': int(n_valid),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2) if not np.isnan(r2) else None,
        'mape': float(mape),
    }

summary_path = os.path.join(Config.FORECAST_DIR, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary: {summary_path}")

print("\n" + "=" * 80)
print("FORECASTING COMPLETE")
print("=" * 80)

if validation_available and rmse is not None:
    print(f"\nValidation Performance:")
    print(f"  RMSE: {rmse:.2f} kWh")
    print(f"  MAE:  {mae:.2f} kWh")
    if not np.isnan(r2):
        print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")