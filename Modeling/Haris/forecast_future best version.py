"""
FUTURE FORECASTING SYSTEM - 7 DAYS AHEAD
=========================================
Erstellt Rolling Forecast für die nächsten 7 Tage
Nutzt trainiertes V13 Model + iterative Predictions
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

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Pfade
    MODEL_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V13_Final"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Forecasts\Future"
    
    MODEL_FILE = os.path.join(MODEL_DIR, "production_model.joblib")
    FEATURES_FILE = os.path.join(MODEL_DIR, "production_features.npy")
    CONFIG_FILE = os.path.join(MODEL_DIR, "production_config.json")
    
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    # Forecast Settings
    FORECAST_DAYS = 7  # 7 Tage in die Zukunft
    FORECAST_INTERVAL = 15  # Minuten (15min Intervall)
    
    PLOT_DPI = 150

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FUTURE FORECASTING SYSTEM - 7 DAYS AHEAD")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n[1/5] Lade trainiertes Modell...")

model = load(Config.MODEL_FILE)
features = np.load(Config.FEATURES_FILE, allow_pickle=True)

with open(Config.CONFIG_FILE, 'r') as f:
    config = json.load(f)

print(f"Modell: {config['model_name']}")
print(f"Features: {len(features)}")

# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("\n[2/5] Lade historische Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

print(f"Letzte Daten: {df.index[-1]}")

# Letzte 30 Tage für Lag-Berechnung
last_30_days = df.tail(30 * 96)  # 96 = 15min Intervalle pro Tag

# ============================================================
# PREPARE FUTURE DATES
# ============================================================

print("\n[3/5] Bereite Zukunfts-Zeitstempel vor...")

# Start: Nach letztem bekannten Zeitpunkt
last_datetime = df.index[-1]
forecast_start = last_datetime + pd.Timedelta(minutes=Config.FORECAST_INTERVAL)

# Ende: 7 Tage später
forecast_end = forecast_start + pd.Timedelta(days=Config.FORECAST_DAYS)

# Erstelle Zeitindex
future_dates = pd.date_range(
    start=forecast_start,
    end=forecast_end,
    freq=f'{Config.FORECAST_INTERVAL}min'
)

print(f"Forecast Start: {future_dates[0]}")
print(f"Forecast Ende:  {future_dates[-1]}")
print(f"Anzahl Steps:   {len(future_dates)}")

# ============================================================
# CREATE FUTURE FEATURES
# ============================================================

print("\n[4/5] Erstelle Future Features...")

# Initialisiere DataFrame für Zukunft
future_df = pd.DataFrame(index=future_dates)

# Zeit-Features erstellen
future_df['Monat'] = future_df.index.month
future_df['Wochentag'] = future_df.index.dayofweek
future_df['Quartal'] = future_df.index.quarter
future_df['Stunde (Lokal)'] = future_df.index.hour
future_df['IstSonntag'] = (future_df.index.dayofweek == 6).astype(int)
future_df['IstArbeitstag'] = (~future_df.index.dayofweek.isin([5, 6])).astype(int)

# Wetter-Features: Nutze saisonale Durchschnitte der letzten 30 Tage
weather_features = [f for f in features if '_lag15' in f]

for wf in weather_features:
    if wf in df.columns:
        # Seasonal average (gleiche Stunde)
        seasonal_avg = df[wf].tail(30 * 96).groupby(df.tail(30 * 96).index.hour).mean()
        future_df[wf] = future_df.index.hour.map(seasonal_avg)
        
        # Falls NaN, nutze Gesamtdurchschnitt
        if future_df[wf].isnull().any():
            future_df[wf].fillna(df[wf].mean(), inplace=True)

print(f"Wetter-Features (seasonal avg): {len(weather_features)}")

# ============================================================
# ROLLING FORECAST
# ============================================================

print("\n[5/5] Erstelle Rolling Forecast...")

# Kombiniere historische + future für Lag-Berechnung
combined_df = pd.concat([df[['Stromverbrauch']], future_df])

predictions = []
prediction_times = []

for i, timestamp in enumerate(future_dates):
    if i % 96 == 0:  # Jeden Tag
        print(f"  Tag {i//96 + 1}/7...")
    
    # Hole Features für diesen Zeitpunkt
    row = future_df.loc[timestamp].copy()
    
    # Lag_24h: Von vor 24 Stunden
    lag_24h_time = timestamp - pd.Timedelta(hours=24)
    
    if lag_24h_time in combined_df.index:
        if lag_24h_time in future_df.index:
            # Nutze Prediction von vor 24h
            idx = list(future_dates).index(lag_24h_time)
            row['Lag_24h'] = predictions[idx]
        else:
            # Nutze historischen Wert
            row['Lag_24h'] = combined_df.loc[lag_24h_time, 'Stromverbrauch']
    else:
        # Fallback: Durchschnitt der Stunde
        hour = timestamp.hour
        row['Lag_24h'] = df['Stromverbrauch'].tail(30*96).groupby(df.tail(30*96).index.hour).mean()[hour]
    
    # Erstelle Feature-Vector in richtiger Reihenfolge
    X_row = []
    for feat in features:
        if feat in row.index:
            X_row.append(row[feat])
        else:
            # Feature fehlt - nutze 0 oder Durchschnitt
            if feat in df.columns:
                X_row.append(df[feat].mean())
            else:
                X_row.append(0)
    
    X_row = np.array(X_row).reshape(1, -1)
    
    # Prediction
    pred = model.predict(X_row)[0]
    
    predictions.append(pred)
    prediction_times.append(timestamp)

print(f"\nForecast abgeschlossen: {len(predictions)} Vorhersagen")

# ============================================================
# SAVE RESULTS
# ============================================================

# DataFrame mit Ergebnissen
results_df = pd.DataFrame({
    'datetime': prediction_times,
    'predicted_consumption': predictions,
    'day': [t.date() for t in prediction_times],
    'hour': [t.hour for t in prediction_times],
    'weekday': [t.strftime('%A') for t in prediction_times],
})

results_path = os.path.join(Config.OUTPUT_DIR, f'forecast_7days_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
results_df.to_csv(results_path, index=False)
print(f"\nErgebnisse gespeichert: {results_path}")

# ============================================================
# PLOTS
# ============================================================

print("\nErstelle Plots...")

# Plot 1: 7-Day Forecast
fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(prediction_times, predictions, linewidth=1.5, color='blue')
ax.set_title('7-Day Rolling Forecast (2025)')
ax.set_xlabel('Datum')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.grid(alpha=0.3)

# Markiere Tage
for i in range(Config.FORECAST_DAYS + 1):
    day_start = prediction_times[0] + pd.Timedelta(days=i)
    ax.axvline(day_start, color='red', linestyle='--', alpha=0.3)
    if i < Config.FORECAST_DAYS:
        ax.text(day_start, ax.get_ylim()[1], f'Tag {i+1}', 
                rotation=90, verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_7days.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 2: Daily Profiles
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for day in range(min(7, Config.FORECAST_DAYS)):
    day_start = prediction_times[0] + pd.Timedelta(days=day)
    day_end = day_start + pd.Timedelta(days=1)
    
    mask = [(t >= day_start and t < day_end) for t in prediction_times]
    day_times = [t for t, m in zip(prediction_times, mask) if m]
    day_preds = [p for p, m in zip(predictions, mask) if m]
    
    if day_preds:
        axes[day].plot(range(len(day_preds)), day_preds, linewidth=1.5)
        axes[day].set_title(f'Tag {day+1}: {day_start.strftime("%Y-%m-%d")}')
        axes[day].set_xlabel('15-min Intervall')
        axes[day].set_ylabel('Stromverbrauch (kWh)')
        axes[day].grid(alpha=0.3)

# Letzter Plot: Zusammenfassung
axes[7].axis('off')
summary_text = f"""
7-Tage Forecast
═══════════════
Start: {prediction_times[0].strftime('%Y-%m-%d %H:%M')}
Ende:  {prediction_times[-1].strftime('%Y-%m-%d %H:%M')}

Predictions: {len(predictions)}

Durchschnitt: {np.mean(predictions):.0f} kWh
Min: {np.min(predictions):.0f} kWh
Max: {np.max(predictions):.0f} kWh

Model: {config['model_name']}
"""
axes[7].text(0.1, 0.9, summary_text, fontsize=12, 
             verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_daily_profiles.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 3: Hourly Average Pattern
fig, ax = plt.subplots(figsize=(14, 6))

hourly_avg = results_df.groupby('hour')['predicted_consumption'].mean()

ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Stunde des Tages')
ax.set_ylabel('Durchschnittlicher Stromverbrauch (kWh)')
ax.set_title('Durchschnittliches Tagesprofil (7-Tage Forecast)')
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'hourly_pattern.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 4: Daily Consumption
fig, ax = plt.subplots(figsize=(12, 6))

daily_consumption = results_df.groupby('day')['predicted_consumption'].sum()

ax.bar(range(len(daily_consumption)), daily_consumption.values, alpha=0.7)
ax.set_xlabel('Tag')
ax.set_ylabel('Tagesverbrauch (kWh)')
ax.set_title('Prognostizierter Tagesverbrauch')
ax.set_xticks(range(len(daily_consumption)))
ax.set_xticklabels([f'Tag {i+1}' for i in range(len(daily_consumption))])
ax.grid(alpha=0.3, axis='y')

# Werte auf Balken
for i, v in enumerate(daily_consumption.values):
    ax.text(i, v, f'{v:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'daily_consumption.png'), dpi=Config.PLOT_DPI)
plt.close()

print(f"Plots gespeichert in: {Config.OUTPUT_DIR}")

# ============================================================
# SUMMARY
# ============================================================

summary = {
    'forecast_created': datetime.now().isoformat(),
    'forecast_period': {
        'start': str(prediction_times[0]),
        'end': str(prediction_times[-1]),
        'days': Config.FORECAST_DAYS,
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
    'daily_totals': {
        f'day_{i+1}': float(v) 
        for i, v in enumerate(daily_consumption.values)
    },
    'method': 'Rolling forecast with Lag_24h and seasonal weather averages',
}

summary_path = os.path.join(Config.OUTPUT_DIR, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary gespeichert: {summary_path}")

# ============================================================
# CONFIDENCE INTERVALS (Optional)
# ============================================================

print("\nBerechne Unsicherheit...")

# Einfache Unsicherheitsschätzung basierend auf Training RMSE
training_rmse = config['test_rmse']

# Unsicherheit steigt mit Forecast-Horizont
uncertainty = []
for i in range(len(predictions)):
    days_ahead = i / 96  # 96 = 15min Intervalle pro Tag
    # Unsicherheit steigt mit sqrt(days)
    conf_interval = training_rmse * (1 + 0.2 * np.sqrt(days_ahead))
    uncertainty.append(conf_interval)

results_df['confidence_interval'] = uncertainty
results_df['lower_bound'] = results_df['predicted_consumption'] - results_df['confidence_interval']
results_df['upper_bound'] = results_df['predicted_consumption'] + results_df['confidence_interval']

# Aktualisiere CSV
results_df.to_csv(results_path, index=False)

# Plot mit Confidence Intervals
fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(prediction_times, predictions, linewidth=2, color='blue', label='Forecast')
ax.fill_between(prediction_times, 
                 results_df['lower_bound'], 
                 results_df['upper_bound'], 
                 alpha=0.3, color='lightblue', label='Confidence Interval')
ax.set_title('7-Day Forecast mit Unsicherheit')
ax.set_xlabel('Datum')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_with_uncertainty.png'), dpi=Config.PLOT_DPI)
plt.close()

print("\n" + "=" * 70)
print("FUTURE FORECASTING ABGESCHLOSSEN")
print("=" * 70)
print(f"\nPrognose für {Config.FORECAST_DAYS} Tage erstellt")
print(f"Durchschnitt: {np.mean(predictions):.0f} kWh")
print(f"Min: {np.min(predictions):.0f} kWh")
print(f"Max: {np.max(predictions):.0f} kWh")
print(f"\nDateien:")
print(f"  CSV:     {results_path}")
print(f"  Plots:   {Config.OUTPUT_DIR}")
print(f"  Summary: {summary_path}")
