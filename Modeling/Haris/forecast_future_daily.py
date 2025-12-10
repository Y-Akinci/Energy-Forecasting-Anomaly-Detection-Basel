"""
DAILY FORECAST WITH API VALIDATION - CORRECTED
===============================================
7-Tage Tages-Verbrauch Prognose mit Echtzeit-Validierung
Nutzt Basel Open Data API für aktuelle Werte
"""

import os
import pandas as pd
import numpy as np
from joblib import load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import requests

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    MODEL_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V14_Daily"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Forecasts\Daily"
    
    MODEL_FILE = os.path.join(MODEL_DIR, "daily_model.joblib")
    FEATURES_FILE = os.path.join(MODEL_DIR, "daily_features.npy")
    CONFIG_FILE = os.path.join(MODEL_DIR, "daily_config.json")
    
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    # API Config - CORRECTED
    API_URL = "https://data.bs.ch/api/explore/v2.1/catalog/datasets/100233/records"
    FORECAST_DAYS = 7
    
    PLOT_DPI = 150

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DAILY FORECAST WITH API VALIDATION")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n[1/6] Lade Model...")

model = load(Config.MODEL_FILE)
features = np.load(Config.FEATURES_FILE, allow_pickle=True)

with open(Config.CONFIG_FILE, 'r') as f:
    config = json.load(f)

print(f"Model: {config['model_name']}")
print(f"Training RMSE: {config['test_rmse']:.0f} kWh")

# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("\n[2/6] Lade historische Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

# Tages-Aggregation
df['date'] = df.index.date
daily_hist = df.groupby('date').agg({
    'Stromverbrauch': 'sum',
    'Monat': 'first',
    'Wochentag': 'first',
    'Quartal': 'first',
    'IstSonntag': 'first',
    'IstArbeitstag': 'first',
}).reset_index()

daily_hist['date'] = pd.to_datetime(daily_hist['date'])
daily_hist = daily_hist.set_index('date').sort_index()

print(f"Historische Daten bis: {daily_hist.index[-1].date()}")

# ============================================================
# FETCH API DATA
# ============================================================

print("\n[3/6] Hole aktuelle Daten von API...")

try:
    # WICHTIG: order_by mit MINUS für descending (neueste zuerst!)
    params = {
        'limit': 3000,  # ~30 Tage × 96 Intervalle
        'order_by': '-timestamp_interval_start'  # Minus = neueste zuerst!
    }
    
    print(f"API Request: {Config.API_URL}")
    print(f"Parameters: limit={params['limit']}, order_by={params['order_by']}")
    
    response = requests.get(Config.API_URL, params=params, timeout=30)
    response.raise_for_status()
    
    api_data = response.json()
    
    if 'results' in api_data and len(api_data['results']) > 0:
        records = api_data['results']
        print(f"API Records erhalten: {len(records)}")
        
        # Zeige ersten Zeitstempel zur Kontrolle
        first_timestamp = records[0].get('timestamp_interval_start', 'unknown')
        print(f"Neuester Zeitstempel in API: {first_timestamp}")
        
        # Parse zu DataFrame
        api_df = pd.DataFrame(records)
        api_df['timestamp'] = pd.to_datetime(api_df['timestamp_interval_start'])
        api_df = api_df.set_index('timestamp').sort_index()
        
        print(f"API Zeitspanne: {api_df.index.min()} bis {api_df.index.max()}")
        
        # Tages-Aggregation
        api_df['date'] = api_df.index.date
        api_daily = api_df.groupby('date').agg({
            'stromverbrauch_kwh': 'sum',
        }).reset_index()
        api_daily['date'] = pd.to_datetime(api_daily['date'])
        api_daily = api_daily.set_index('date').sort_index()
        api_daily = api_daily.rename(columns={'stromverbrauch_kwh': 'actual_consumption'})
        
        print(f"API Tages-Daten: {len(api_daily)} Tage")
        print(f"Neueste Tages-Daten: {api_daily.index[-1].date()}")
        
        api_available = True
    else:
        print("WARNUNG: Keine Daten in API Response")
        api_available = False
        api_daily = None

except requests.exceptions.RequestException as e:
    print(f"API Request Fehler: {e}")
    print("Fortfahren ohne API-Validierung...")
    api_available = False
    api_daily = None
except Exception as e:
    print(f"Unerwarteter Fehler: {e}")
    print("Fortfahren ohne API-Validierung...")
    api_available = False
    api_daily = None

# ============================================================
# PREPARE FUTURE DATES
# ============================================================

print("\n[4/6] Bereite Zukunfts-Daten vor...")

last_date = daily_hist.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=Config.FORECAST_DAYS,
    freq='D'
)

print(f"Forecast: {future_dates[0].date()} bis {future_dates[-1].date()}")

# Erstelle Future DataFrame
future_df = pd.DataFrame(index=future_dates)
future_df['Monat'] = future_df.index.month
future_df['Wochentag'] = future_df.index.dayofweek
future_df['Quartal'] = future_df.index.quarter
future_df['IstSonntag'] = (future_df.index.dayofweek == 6).astype(int)
future_df['IstArbeitstag'] = (~future_df.index.dayofweek.isin([5, 6])).astype(int)

# Kombiniere historical + future
combined = pd.concat([daily_hist[['Stromverbrauch']], future_df])

# ============================================================
# ROLLING FORECAST
# ============================================================

print("\n[5/6] Erstelle Rolling Forecast...")

predictions = []
prediction_dates = []

for i, date in enumerate(future_dates):
    print(f"  Forecasting Tag {i+1}/7: {date.date()}...")
    
    # Lag Features
    lag_1day = combined.loc[date - pd.Timedelta(days=1), 'Stromverbrauch'] if (date - pd.Timedelta(days=1)) in combined.index else np.nan
    lag_7days = combined.loc[date - pd.Timedelta(days=7), 'Stromverbrauch'] if (date - pd.Timedelta(days=7)) in combined.index else np.nan
    lag_14days = combined.loc[date - pd.Timedelta(days=14), 'Stromverbrauch'] if (date - pd.Timedelta(days=14)) in combined.index else np.nan
    
    # Rolling averages (letzte 7/14 Tage)
    recent_data = combined.loc[:date - pd.Timedelta(days=1), 'Stromverbrauch']
    rolling_7d = recent_data.tail(7).mean() if len(recent_data) >= 7 else np.nan
    rolling_14d = recent_data.tail(14).mean() if len(recent_data) >= 14 else np.nan
    
    # Weekday average (letzter Monat)
    weekday = date.dayofweek
    weekday_data = daily_hist[daily_hist.index.dayofweek == weekday]['Stromverbrauch'].tail(4)
    weekday_avg = weekday_data.mean() if len(weekday_data) > 0 else np.nan
    
    # Feature Row
    row = {
        'Lag_1day': lag_1day,
        'Lag_7days': lag_7days,
        'Lag_14days': lag_14days,
        'Rolling_7d_mean': rolling_7d,
        'Rolling_14d_mean': rolling_14d,
        'Weekday_avg': weekday_avg,
        'Monat': date.month,
        'Wochentag': weekday,
        'Quartal': date.quarter,
        'IstSonntag': int(weekday == 6),
        'IstArbeitstag': int(weekday not in [5, 6]),
    }
    
    # Check for NaN
    if any(pd.isna(v) for k, v in row.items() if k in features):
        print(f"    WARNUNG: NaN in Features, nutze Fallback")
        # Fallback: Durchschnitt der letzten Woche
        row['Lag_1day'] = recent_data.tail(7).mean()
        row['Lag_7days'] = recent_data.tail(14).mean()
        row['Lag_14days'] = recent_data.tail(21).mean()
        row['Rolling_7d_mean'] = recent_data.tail(7).mean()
        row['Rolling_14d_mean'] = recent_data.tail(14).mean()
        row['Weekday_avg'] = recent_data.tail(30).mean()
    
    # Erstelle Feature Vector
    X_row = np.array([row[f] for f in features]).reshape(1, -1)
    
    # Predict
    pred = model.predict(X_row)[0]
    
    predictions.append(pred)
    prediction_dates.append(date)
    
    # Update combined für nächste Iteration
    combined.loc[date, 'Stromverbrauch'] = pred

print(f"Forecast fertig: {len(predictions)} Tage")

# ============================================================
# COMPARE WITH API
# ============================================================

results_df = pd.DataFrame({
    'date': prediction_dates,
    'predicted': predictions,
})
results_df = results_df.set_index('date')

if api_available and api_daily is not None:
    print("\n[6/6] Vergleiche mit API Daten...")
    
    # Merge mit API Daten
    comparison = results_df.join(api_daily, how='left')
    
    # Berechne Metriken für verfügbare Daten
    valid_mask = comparison['actual_consumption'].notna()
    
    if valid_mask.sum() > 0:
        valid_comparison = comparison[valid_mask]
        
        rmse = np.sqrt(np.mean((valid_comparison['actual_consumption'] - valid_comparison['predicted'])**2))
        mae = np.mean(np.abs(valid_comparison['actual_consumption'] - valid_comparison['predicted']))
        
        # R² nur wenn mehr als 1 Wert
        if len(valid_comparison) > 1:
            from sklearn.metrics import r2_score
            r2 = r2_score(valid_comparison['actual_consumption'], valid_comparison['predicted'])
        else:
            r2 = np.nan
        
        print(f"\nAPI Validation ({valid_mask.sum()} Tage verfügbar):")
        print(f"  RMSE: {rmse:.0f} kWh")
        print(f"  MAE:  {mae:.0f} kWh")
        if not np.isnan(r2):
            print(f"  R²:   {r2:.4f}")
        
        comparison['error'] = comparison['actual_consumption'] - comparison['predicted']
        comparison['abs_error'] = np.abs(comparison['error'])
        
        results_df = comparison
    else:
        print("\nKeine überlappenden Daten für Validation gefunden")
        results_df['actual_consumption'] = np.nan
        results_df['error'] = np.nan
        valid_mask = pd.Series(False, index=results_df.index)
else:
    print("\n[6/6] Keine API Validierung möglich")
    results_df['actual_consumption'] = np.nan
    results_df['error'] = np.nan
    valid_mask = pd.Series(False, index=results_df.index)

# ============================================================
# SAVE
# ============================================================

csv_path = os.path.join(Config.OUTPUT_DIR, f'daily_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
results_df.to_csv(csv_path)
print(f"\nErgebnisse: {csv_path}")

# ============================================================
# PLOTS
# ============================================================

print("\nErstelle Plots...")

# Plot 1: Forecast mit API Validation
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(results_df.index, results_df['predicted'], 
        marker='o', linewidth=2, markersize=8, label='Forecast', color='blue')

if api_available and 'actual_consumption' in results_df.columns:
    valid = results_df['actual_consumption'].notna()
    if valid.sum() > 0:
        ax.plot(results_df.index[valid], results_df.loc[valid, 'actual_consumption'],
                marker='s', linewidth=2, markersize=8, label='Actual (API)', color='green')

ax.set_title('7-Day Daily Consumption Forecast with API Validation')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Consumption (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_validation.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 2: Error Analysis (if API available)
if api_available and 'error' in results_df.columns and results_df['error'].notna().sum() > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    valid = results_df['error'].notna()
    
    # Error timeline
    axes[0].bar(results_df.index[valid], results_df.loc[valid, 'error'], alpha=0.7)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Prediction Error')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Error (kWh)')
    axes[0].grid(alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    
    # Scatter
    axes[1].scatter(results_df.loc[valid, 'actual_consumption'],
                   results_df.loc[valid, 'predicted'], s=100, alpha=0.7)
    
    min_val = min(results_df.loc[valid, 'actual_consumption'].min(),
                  results_df.loc[valid, 'predicted'].min())
    max_val = max(results_df.loc[valid, 'actual_consumption'].max(),
                  results_df.loc[valid, 'predicted'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1].set_xlabel('Actual (kWh)')
    axes[1].set_ylabel('Predicted (kWh)')
    axes[1].set_title('Prediction vs Actual')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'error_analysis.png'), dpi=Config.PLOT_DPI)
    plt.close()

# Plot 3: Summary
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

summary_text = f"""
DAILY FORECAST SUMMARY
══════════════════════

Forecast Period: {prediction_dates[0].date()} - {prediction_dates[-1].date()}
Days: {len(predictions)}

Predicted Consumption:
  Total: {sum(predictions):.0f} kWh
  Average: {np.mean(predictions):.0f} kWh/day
  Min: {min(predictions):.0f} kWh
  Max: {max(predictions):.0f} kWh

Model: {config['model_name']}
Training RMSE: {config['test_rmse']:.0f} kWh
"""

if api_available and valid_mask.sum() > 0:
    summary_text += f"""
API Validation ({valid_mask.sum()} days):
  RMSE: {rmse:.0f} kWh
  MAE: {mae:.0f} kWh"""
    if not np.isnan(r2):
        summary_text += f"""
  R²: {r2:.4f}"""

ax.text(0.1, 0.9, summary_text, fontsize=12,
        verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'summary.png'), dpi=Config.PLOT_DPI)
plt.close()

print(f"Plots: {Config.OUTPUT_DIR}")

# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    'created': datetime.now().isoformat(),
    'forecast_period': {
        'start': str(prediction_dates[0].date()),
        'end': str(prediction_dates[-1].date()),
        'days': len(predictions),
    },
    'predictions': {
        'total_kwh': float(sum(predictions)),
        'avg_kwh': float(np.mean(predictions)),
        'min_kwh': float(min(predictions)),
        'max_kwh': float(max(predictions)),
    },
    'model': {
        'name': config['model_name'],
        'training_rmse': config['test_rmse'],
    },
}

if api_available and valid_mask.sum() > 0:
    summary['api_validation'] = {
        'days': int(valid_mask.sum()),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2) if not np.isnan(r2) else None,
    }

summary_path = os.path.join(Config.OUTPUT_DIR, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary: {summary_path}")

print("\n" + "=" * 70)
print("DAILY FORECAST FERTIG")
print("=" * 70)
