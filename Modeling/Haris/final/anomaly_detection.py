"""
ANOMALY DETECTION SYSTEM
=========================
Erkennt Anomalien in historischen und prognostizierten Daten
Methoden: Z-Score, IQR, Isolation Forest, Prediction Intervals
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Config:
    BASE_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\Final"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    
    ANOMALY_DIR = os.path.join(BASE_DIR, "anomalies")
    PLOTS_DIR = os.path.join(BASE_DIR, "anomaly_plots")
    
    for dir in [ANOMALY_DIR, PLOTS_DIR]:
        os.makedirs(dir, exist_ok=True)
    
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    ZSCORE_THRESHOLD = 3.0
    IQR_MULTIPLIER = 1.5
    CONTAMINATION = 0.01
    
    PLOT_DPI = 150

print("=" * 80)
print("ANOMALY DETECTION SYSTEM")
print("=" * 80)

print("\n[1/5] Lade historische Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

consumption = df['Stromverbrauch'].copy()

print(f"Zeitraum: {df.index.min()} bis {df.index.max()}")
print(f"Samples: {len(df):,}")

print("\n[2/5] Z-Score Methode...")

mean_consumption = consumption.mean()
std_consumption = consumption.std()

z_scores = np.abs((consumption - mean_consumption) / std_consumption)
z_anomalies = z_scores > Config.ZSCORE_THRESHOLD

n_z_anomalies = z_anomalies.sum()
pct_z_anomalies = (n_z_anomalies / len(consumption)) * 100

print(f"Z-Score Threshold: {Config.ZSCORE_THRESHOLD}")
print(f"Anomalien gefunden: {n_z_anomalies:,} ({pct_z_anomalies:.2f}%)")

print("\n[3/5] IQR Methode...")

Q1 = consumption.quantile(0.25)
Q3 = consumption.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - Config.IQR_MULTIPLIER * IQR
upper_bound = Q3 + Config.IQR_MULTIPLIER * IQR

iqr_anomalies = (consumption < lower_bound) | (consumption > upper_bound)

n_iqr_anomalies = iqr_anomalies.sum()
pct_iqr_anomalies = (n_iqr_anomalies / len(consumption)) * 100

print(f"IQR Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Anomalien gefunden: {n_iqr_anomalies:,} ({pct_iqr_anomalies:.2f}%)")

print("\n[4/5] Isolation Forest...")

X_iso = consumption.values.reshape(-1, 1)

iso_forest = IsolationForest(
    contamination=Config.CONTAMINATION,
    random_state=42,
    n_jobs=-1
)

iso_predictions = iso_forest.fit_predict(X_iso)
iso_anomalies = iso_predictions == -1

n_iso_anomalies = iso_anomalies.sum()
pct_iso_anomalies = (n_iso_anomalies / len(consumption)) * 100

print(f"Contamination: {Config.CONTAMINATION}")
print(f"Anomalien gefunden: {n_iso_anomalies:,} ({pct_iso_anomalies:.2f}%)")

print("\n[5/5] Kombinierte Anomalien...")

combined_anomalies = z_anomalies | iqr_anomalies | iso_anomalies

n_combined = combined_anomalies.sum()
pct_combined = (n_combined / len(consumption)) * 100

print(f"Gesamt Anomalien: {n_combined:,} ({pct_combined:.2f}%)")

anomalies_df = pd.DataFrame({
    'datetime': df.index,
    'consumption': consumption.values,
    'z_score': z_scores.values,
    'z_anomaly': z_anomalies.values,
    'iqr_anomaly': iqr_anomalies.values,
    'iso_anomaly': iso_anomalies,
    'combined_anomaly': combined_anomalies.values,
})
anomalies_df = anomalies_df.set_index('datetime')

anomalies_df['hour'] = anomalies_df.index.hour
anomalies_df['weekday'] = anomalies_df.index.dayofweek
anomalies_df['month'] = anomalies_df.index.month

anomalies_only = anomalies_df[anomalies_df['combined_anomaly']]

anomaly_path = os.path.join(Config.ANOMALY_DIR, f'historical_anomalies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
anomalies_only.to_csv(anomaly_path)
print(f"\nAnomaly CSV: {anomaly_path}")

print("\nErstelle Plots...")

fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(consumption.index, consumption.values, linewidth=0.5, alpha=0.7, label='Normal', color='blue')

anomaly_times = consumption.index[combined_anomalies]
anomaly_values = consumption.values[combined_anomalies]
ax.scatter(anomaly_times, anomaly_values, color='red', s=20, alpha=0.8, label='Anomaly', zorder=5)

ax.set_title('Stromverbrauch mit Anomalien (Kombinierte Methoden)')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'anomalies_timeline.png'), dpi=Config.PLOT_DPI)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].hist(consumption.values, bins=100, alpha=0.7, edgecolor='black', color='blue')
axes[0, 0].axvline(lower_bound, color='red', linestyle='--', label='IQR Lower')
axes[0, 0].axvline(upper_bound, color='red', linestyle='--', label='IQR Upper')
axes[0, 0].set_xlabel('Stromverbrauch (kWh)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Consumption Distribution mit IQR Bounds')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

axes[0, 1].scatter(consumption.values, z_scores.values, alpha=0.3, s=1)
axes[0, 1].axhline(Config.ZSCORE_THRESHOLD, color='red', linestyle='--', label='Threshold')
axes[0, 1].axhline(-Config.ZSCORE_THRESHOLD, color='red', linestyle='--')
axes[0, 1].set_xlabel('Stromverbrauch (kWh)')
axes[0, 1].set_ylabel('Z-Score')
axes[0, 1].set_title('Z-Score Analysis')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

method_counts = {
    'Z-Score': n_z_anomalies,
    'IQR': n_iqr_anomalies,
    'Isolation Forest': n_iso_anomalies,
    'Combined': n_combined
}

axes[1, 0].bar(method_counts.keys(), method_counts.values(), alpha=0.7, color=['blue', 'green', 'orange', 'red'])
axes[1, 0].set_ylabel('Anzahl Anomalien')
axes[1, 0].set_title('Anomalien nach Methode')
axes[1, 0].set_xticklabels(method_counts.keys(), rotation=45, ha='right')
axes[1, 0].grid(alpha=0.3, axis='y')

for i, (k, v) in enumerate(method_counts.items()):
    axes[1, 0].text(i, v, f'{v:,}', ha='center', va='bottom')

hourly_anomalies = anomalies_only.groupby('hour').size()
axes[1, 1].bar(hourly_anomalies.index, hourly_anomalies.values, alpha=0.7, color='red')
axes[1, 1].set_xlabel('Stunde')
axes[1, 1].set_ylabel('Anzahl Anomalien')
axes[1, 1].set_title('Anomalien nach Stunde')
axes[1, 1].set_xticks(range(0, 24))
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'anomaly_analysis.png'), dpi=Config.PLOT_DPI)
plt.close()

last_30_days = consumption.tail(30 * 96)
last_30_anomalies = combined_anomalies.tail(30 * 96)

fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(last_30_days.index, last_30_days.values, linewidth=1, alpha=0.7, color='blue')

if last_30_anomalies.sum() > 0:
    anomaly_times_30 = last_30_days.index[last_30_anomalies]
    anomaly_values_30 = last_30_days.values[last_30_anomalies]
    ax.scatter(anomaly_times_30, anomaly_values_30, color='red', s=30, alpha=0.8, label='Anomaly', zorder=5)

ax.set_title('Anomalien - Letzte 30 Tage')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'anomalies_last30days.png'), dpi=Config.PLOT_DPI)
plt.close()

print(f"Plots: {Config.PLOTS_DIR}")

summary = {
    'analysis_date': datetime.now().isoformat(),
    'data_period': {
        'start': str(df.index.min()),
        'end': str(df.index.max()),
        'n_samples': len(df),
    },
    'methods': {
        'z_score': {
            'threshold': Config.ZSCORE_THRESHOLD,
            'n_anomalies': int(n_z_anomalies),
            'percentage': float(pct_z_anomalies),
        },
        'iqr': {
            'multiplier': Config.IQR_MULTIPLIER,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'n_anomalies': int(n_iqr_anomalies),
            'percentage': float(pct_iqr_anomalies),
        },
        'isolation_forest': {
            'contamination': Config.CONTAMINATION,
            'n_anomalies': int(n_iso_anomalies),
            'percentage': float(pct_iso_anomalies),
        },
        'combined': {
            'n_anomalies': int(n_combined),
            'percentage': float(pct_combined),
        },
    },
    'statistics': {
        'mean': float(mean_consumption),
        'std': float(std_consumption),
        'min': float(consumption.min()),
        'max': float(consumption.max()),
        'Q1': float(Q1),
        'Q3': float(Q3),
        'IQR': float(IQR),
    },
}

summary_path = os.path.join(Config.ANOMALY_DIR, f'anomaly_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary: {summary_path}")

print("\n" + "=" * 80)
print("ANOMALY DETECTION FOR HISTORICAL DATA COMPLETE")
print("=" * 80)

print("\nFUTURE FORECAST ANOMALY DETECTION")
print("=" * 80)

try:
    forecast_files = [f for f in os.listdir(os.path.join(Config.BASE_DIR, "forecasts")) if f.startswith('forecast_') and f.endswith('.csv')]
    
    if forecast_files:
        latest_forecast = sorted(forecast_files)[-1]
        forecast_path = os.path.join(Config.BASE_DIR, "forecasts", latest_forecast)
        
        print(f"\nLade Forecast: {latest_forecast}")
        
        forecast_df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
        
        if 'predicted' in forecast_df.columns:
            forecast_consumption = forecast_df['predicted']
            
            forecast_z_scores = np.abs((forecast_consumption - mean_consumption) / std_consumption)
            forecast_z_anomalies = forecast_z_scores > Config.ZSCORE_THRESHOLD
            
            forecast_iqr_anomalies = (forecast_consumption < lower_bound) | (forecast_consumption > upper_bound)
            
            forecast_combined = forecast_z_anomalies | forecast_iqr_anomalies
            
            n_forecast_anomalies = forecast_combined.sum()
            pct_forecast_anomalies = (n_forecast_anomalies / len(forecast_consumption)) * 100
            
            print(f"\nForecast Anomalien: {n_forecast_anomalies:,} ({pct_forecast_anomalies:.2f}%)")
            
            forecast_anomalies_df = pd.DataFrame({
                'datetime': forecast_df.index,
                'predicted': forecast_consumption.values,
                'z_score': forecast_z_scores.values,
                'z_anomaly': forecast_z_anomalies.values,
                'iqr_anomaly': forecast_iqr_anomalies.values,
                'combined_anomaly': forecast_combined.values,
            })
            forecast_anomalies_df = forecast_anomalies_df.set_index('datetime')
            
            forecast_anomaly_path = os.path.join(Config.ANOMALY_DIR, f'forecast_anomalies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            forecast_anomalies_df[forecast_anomalies_df['combined_anomaly']].to_csv(forecast_anomaly_path)
            
            if 'actual_consumption' in forecast_df.columns:
                valid_mask = forecast_df['actual_consumption'].notna()
                
                if valid_mask.sum() > 0:
                    residuals = forecast_df.loc[valid_mask, 'actual_consumption'] - forecast_df.loc[valid_mask, 'predicted']
                    
                    residual_mean = residuals.mean()
                    residual_std = residuals.std()
                    
                    residual_z_scores = np.abs((residuals - residual_mean) / residual_std)
                    residual_anomalies = residual_z_scores > Config.ZSCORE_THRESHOLD
                    
                    n_residual_anomalies = residual_anomalies.sum()
                    
                    print(f"Residual Anomalien: {n_residual_anomalies} (große Vorhersagefehler)")
            
            fig, ax = plt.subplots(figsize=(20, 6))
            
            ax.plot(forecast_consumption.index, forecast_consumption.values, linewidth=1, alpha=0.7, color='blue', label='Forecast')
            
            if forecast_combined.sum() > 0:
                anomaly_times = forecast_consumption.index[forecast_combined]
                anomaly_values = forecast_consumption.values[forecast_combined]
                ax.scatter(anomaly_times, anomaly_values, color='red', s=30, alpha=0.8, label='Anomaly', zorder=5)
            
            ax.axhline(lower_bound, color='orange', linestyle='--', alpha=0.5, label='IQR Bounds')
            ax.axhline(upper_bound, color='orange', linestyle='--', alpha=0.5)
            
            ax.set_title('Forecast mit Anomalien')
            ax.set_xlabel('Zeit')
            ax.set_ylabel('Stromverbrauch (kWh)')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(Config.PLOTS_DIR, 'forecast_anomalies.png'), dpi=Config.PLOT_DPI)
            plt.close()
            
            print(f"\nForecast Anomaly CSV: {forecast_anomaly_path}")
            print(f"Forecast Anomaly Plot: {Config.PLOTS_DIR}")
        
        print("\n" + "=" * 80)
        print("ANOMALY DETECTION COMPLETE")
        print("=" * 80)
    else:
        print("\nKeine Forecast Files gefunden")
        
except Exception as e:
    print(f"\nFehler bei Future Anomaly Detection: {e}")
