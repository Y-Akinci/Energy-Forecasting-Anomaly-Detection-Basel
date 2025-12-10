"""
MODELING FRAMEWORK V14 - TAGES-VERBRAUCH PROGNOSE
==================================================
Predictet Gesamt-Stromverbrauch für den nächsten Tag
Target: Summe von 96 x 15-min Intervallen (24h)
Keine kurzen Lags - realistisches R²
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump
import json
from datetime import datetime

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V14_Daily"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    START_DATE = "2015-01-01 00:00:00"
    END_DATE = "2024-12-31 23:45:00"
    
    TRAIN_RATIO = 0.7
    CV_SPLITS = 5
    
    # Tages-Features (keine kurzen Lags!)
    SELECTED_FEATURES = [
        'Monat',
        'Wochentag',
        'Quartal',
        'IstSonntag',
        'IstArbeitstag',
    ]
    
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
    }
    
    GB_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'subsample': 0.8,
        'random_state': 42,
    }
    
    XGB_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    RIDGE_PARAMS = {
        'alpha': 10.0,
        'random_state': 42,
    }
    
    PLOT_DPI = 150

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("V14 - TAGES-VERBRAUCH PROGNOSE")
print("=" * 70)

# ============================================================
# DATEN LADEN & AGGREGIEREN
# ============================================================

print("\n[1/5] Lade und aggregiere Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

df = df.loc[Config.START_DATE:Config.END_DATE].copy()
print(f"15-min Daten: {len(df)} Zeilen")

# Erstelle Tages-Aggregation
df['date'] = df.index.date

daily_df = df.groupby('date').agg({
    'Stromverbrauch': 'sum',  # Tages-Summe
    'Monat': 'first',
    'Wochentag': 'first',
    'Quartal': 'first',
    'IstSonntag': 'first',
    'IstArbeitstag': 'first',
}).reset_index()

daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.set_index('date').sort_index()

print(f"Tages-Daten: {len(daily_df)} Tage")
print(f"Durchschnitt: {daily_df['Stromverbrauch'].mean():.0f} kWh/Tag")

# ============================================================
# FEATURES & TARGET
# ============================================================

print("\n[2/5] Erstelle Features & Target...")

# Lag-Features auf Tages-Ebene
daily_df['Lag_1day'] = daily_df['Stromverbrauch'].shift(1)
daily_df['Lag_7days'] = daily_df['Stromverbrauch'].shift(7)
daily_df['Lag_14days'] = daily_df['Stromverbrauch'].shift(14)

# Rolling Durchschnitte
daily_df['Rolling_7d_mean'] = daily_df['Stromverbrauch'].shift(1).rolling(7).mean()
daily_df['Rolling_14d_mean'] = daily_df['Stromverbrauch'].shift(1).rolling(14).mean()

# Wochentags-Durchschnitt (letzter Monat)
daily_df['Weekday_avg'] = daily_df.groupby('Wochentag')['Stromverbrauch'].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean()
)

print("Lag-Features erstellt")

# Target: Morgen
y = daily_df['Stromverbrauch'].shift(-1)

# Features
feature_cols = [
    'Lag_1day',
    'Lag_7days', 
    'Lag_14days',
    'Rolling_7d_mean',
    'Rolling_14d_mean',
    'Weekday_avg',
    'Monat',
    'Wochentag',
    'Quartal',
    'IstSonntag',
    'IstArbeitstag',
]

X = daily_df[feature_cols].copy()

# Bereinigen
mask = y.notna() & X.notna().all(axis=1)
X = X.loc[mask]
y = y.loc[mask]

print(f"Samples nach Bereinigung: {len(X)}")
print(f"Features: {len(feature_cols)}")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

print("\n[3/5] Train/Test Split...")

n_train = int(len(X) * Config.TRAIN_RATIO)

X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

print(f"Train: {len(X_train)} Tage")
print(f"Test:  {len(X_test)} Tage")

# ============================================================
# BASELINE
# ============================================================

print("\n[4/5] Baseline...")

baseline_rmse = np.sqrt(mean_squared_error(y_test, X_test['Lag_1day']))
baseline_r2 = r2_score(y_test, X_test['Lag_1day'])
print(f"Baseline (Lag_1day): RMSE={baseline_rmse:.0f} kWh, R²={baseline_r2:.4f}")

# ============================================================
# MODEL TRAINING
# ============================================================

print("\n[5/5] Trainiere Modelle...")

results = []

def train_and_evaluate(model, name, params):
    print(f"\n{name}:")
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Train RMSE: {train_rmse:.0f} kWh, R²: {train_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.0f} kWh, R²: {test_r2:.4f}")
    print(f"  Overfitting: {(train_r2 - test_r2):.4f}")
    
    # CV
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    cv_rmse = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        m = model.__class__(**params)
        m.fit(X_tr, y_tr)
        cv_rmse.append(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))
    
    cv_rmse_mean = np.mean(cv_rmse)
    print(f"  CV RMSE: {cv_rmse_mean:.0f} kWh")
    
    return {
        'name': name,
        'model': model,
        'params': params,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
    }

# Train
rf = RandomForestRegressor(**Config.RF_PARAMS)
results.append(train_and_evaluate(rf, "RandomForest", Config.RF_PARAMS))

gb = GradientBoostingRegressor(**Config.GB_PARAMS)
results.append(train_and_evaluate(gb, "GradientBoosting", Config.GB_PARAMS))

if HAS_XGB:
    xgb_model = xgb.XGBRegressor(**Config.XGB_PARAMS)
    results.append(train_and_evaluate(xgb_model, "XGBoost", Config.XGB_PARAMS))

ridge = Ridge(**Config.RIDGE_PARAMS)
results.append(train_and_evaluate(ridge, "Ridge", Config.RIDGE_PARAMS))

# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

best = min(results, key=lambda x: x['test_rmse'])

for r in results:
    print(f"{r['name']:20s} RMSE: {r['test_rmse']:7.0f} kWh, R²: {r['test_r2']:.4f}")

print(f"\nBEST: {best['name']}")
print(f"RMSE: {best['test_rmse']:.0f} kWh")
print(f"R²:   {best['test_r2']:.4f}")
print(f"Baseline Verbesserung: {((baseline_rmse - best['test_rmse'])/baseline_rmse*100):.1f}%")

# ============================================================
# PLOTS
# ============================================================

# Forecast Timeline
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(y_test.index, y_test.values, label='True', linewidth=1.5)
ax.plot(y_test.index, best['y_test_pred'], label='Predicted', linewidth=1.5, linestyle='--')
ax.set_title(f'Daily Consumption Forecast - {best["name"]}')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Consumption (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_timeline.png'), dpi=Config.PLOT_DPI)
plt.close()

# Scatter
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(y_test.values, best['y_test_pred'], alpha=0.5, s=20)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('True Daily Consumption (kWh)')
ax.set_ylabel('Predicted Daily Consumption (kWh)')
ax.set_title(f'Prediction Quality - {best["name"]}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'scatter.png'), dpi=Config.PLOT_DPI)
plt.close()

# Feature Importance
if hasattr(best['model'], 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    ax.barh(range(len(importances)), importances['importance'])
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {best["name"]}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'feature_importance.png'), dpi=Config.PLOT_DPI)
    plt.close()

# ============================================================
# SAVE
# ============================================================

dump(best['model'], os.path.join(Config.OUTPUT_DIR, 'daily_model.joblib'))
np.save(os.path.join(Config.OUTPUT_DIR, 'daily_features.npy'), X_train.columns.to_numpy())

config_data = {
    'model_name': best['name'],
    'test_rmse': float(best['test_rmse']),
    'test_r2': float(best['test_r2']),
    'features': feature_cols,
    'params': best['params'],
    'forecast_type': 'daily_consumption',
    'trained_date': datetime.now().isoformat(),
}

with open(os.path.join(Config.OUTPUT_DIR, 'daily_config.json'), 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\nModel gespeichert: {Config.OUTPUT_DIR}")
print("\n" + "=" * 70)
print("V14 FERTIG - TAGES-PROGNOSE MODEL")
print("=" * 70)
