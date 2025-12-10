"""
MODELING FRAMEWORK V17 - 15-MIN PROGNOSE MIT TAGES-FEATURES
============================================================
15-min Forecast mit aggregierten Tages-Features
Nutzt nur: Lag_1day, Lag_7days, Rolling means, etc.
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
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V17_15min"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    START_DATE = "2015-01-01 00:00:00"
    END_DATE = "2024-12-31 23:45:00"
    
    # 15-min Forecast
    FORECAST_HORIZON_STEPS = 1  # 1 step = 15min
    FORECAST_HORIZON_NAME = "15min"
    
    TRAIN_RATIO = 0.7
    CV_SPLITS = 5
    
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
print("V17 - 15-MIN PROGNOSE MIT TAGES-FEATURES")
print("=" * 70)

# ============================================================
# DATEN LADEN
# ============================================================

print("\n[1/6] Lade Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

df = df.loc[Config.START_DATE:Config.END_DATE].copy()
print(f"15-min Daten: {len(df):,} Zeilen")

# ============================================================
# TAGES-AGGREGATION FÜR FEATURES
# ============================================================

print("\n[2/6] Erstelle Tages-Features...")

# Erstelle Tages-Aggregation für Features
df['date'] = df.index.date

daily_agg = df.groupby('date').agg({
    'Stromverbrauch': 'sum',  # Tages-Summe
    'Monat': 'first',
    'Wochentag': 'first',
    'Quartal': 'first',
    'IstSonntag': 'first',
    'IstArbeitstag': 'first',
}).reset_index()

daily_agg['date'] = pd.to_datetime(daily_agg['date'])
daily_agg = daily_agg.set_index('date').sort_index()

# Tages-Lag Features
daily_agg['Lag_1day'] = daily_agg['Stromverbrauch'].shift(1)
daily_agg['Lag_7days'] = daily_agg['Stromverbrauch'].shift(7)
daily_agg['Lag_14days'] = daily_agg['Stromverbrauch'].shift(14)

# Rolling Durchschnitte
daily_agg['Rolling_7d_mean'] = daily_agg['Stromverbrauch'].shift(1).rolling(7).mean()
daily_agg['Rolling_14d_mean'] = daily_agg['Stromverbrauch'].shift(1).rolling(14).mean()

# Wochentags-Durchschnitt
daily_agg['Weekday_avg'] = daily_agg.groupby('Wochentag')['Stromverbrauch'].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean()
)

print("Tages-Features erstellt")

# Merge zurück zu 15-min Daten
df = df.merge(
    daily_agg[['Lag_1day', 'Lag_7days', 'Lag_14days', 'Rolling_7d_mean', 'Rolling_14d_mean', 'Weekday_avg']],
    left_on='date',
    right_index=True,
    how='left'
)

print("Tages-Features zu 15-min Daten gemerged")

# ============================================================
# TARGET & FEATURES
# ============================================================

print("\n[3/6] Target & Features...")

# Target: 15min voraus
target_col = "Stromverbrauch"
y = df[target_col].shift(-Config.FORECAST_HORIZON_STEPS).astype(float)

# Features - GENAU diese!
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

print(f"\nFeatures ({len(feature_cols)}):")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {feat}")

X = df[feature_cols].copy()

# NaNs bereinigen
mask = y.notna() & X.notna().all(axis=1)
X = X.loc[mask]
y = y.loc[mask]

print(f"\nNach Bereinigung:")
print(f"  Samples: {len(X):,}")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

print("\n[4/6] Train/Test Split...")

n_train = int(len(X) * Config.TRAIN_RATIO)

X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

print(f"Train: {len(X_train):,} samples")
print(f"Test:  {len(X_test):,} samples")

# ============================================================
# BASELINE
# ============================================================

print("\n[5/6] Baseline...")

baseline_rmse = np.sqrt(mean_squared_error(y_test, X_test['Lag_1day']))
baseline_r2 = r2_score(y_test, X_test['Lag_1day'])
print(f"Baseline (Lag_1day): RMSE={baseline_rmse:.2f} kWh, R²={baseline_r2:.4f}")

# ============================================================
# MODEL TRAINING
# ============================================================

print("\n[6/6] Trainiere Modelle...")

results = []

def train_and_evaluate(model, name, params):
    print(f"\n{name}:")
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Train RMSE: {train_rmse:.2f} kWh, R²: {train_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.2f} kWh, R²: {test_r2:.4f}")
    
    overfitting = train_r2 - test_r2
    print(f"  Overfitting: {overfitting:.4f}")
    
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
    cv_rmse_std = np.std(cv_rmse)
    print(f"  CV RMSE: {cv_rmse_mean:.2f} ± {cv_rmse_std:.2f} kWh")
    
    return {
        'name': name,
        'model': model,
        'params': params,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting': overfitting,
        'cv_rmse_mean': cv_rmse_mean,
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
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS - 15-MIN FORECAST MIT TAGES-FEATURES")
print("=" * 70)

comparison = []
for r in results:
    comparison.append({
        'Model': r['name'],
        'Test_RMSE': r['test_rmse'],
        'Test_MAE': r['test_mae'],
        'Test_R2': r['test_r2'],
        'Overfitting': r['overfitting'],
        'CV_RMSE': r['cv_rmse_mean'],
    })

df_comparison = pd.DataFrame(comparison).sort_values('Test_RMSE')
print(df_comparison.to_string(index=False))

best = min(results, key=lambda x: x['test_rmse'])

print(f"\nBEST: {best['name']}")
print(f"RMSE: {best['test_rmse']:.2f} kWh")
print(f"MAE:  {best['test_mae']:.2f} kWh")
print(f"R²:   {best['test_r2']:.4f}")

improvement = (baseline_rmse - best['test_rmse']) / baseline_rmse * 100
print(f"\nBaseline RMSE: {baseline_rmse:.2f} kWh")
print(f"Best RMSE:     {best['test_rmse']:.2f} kWh")
print(f"Verbesserung:  {improvement:.1f}%")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

if hasattr(best['model'], 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Ranking:")
    for i, row in importances.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # Save
    importance_path = os.path.join(Config.OUTPUT_DIR, 'feature_importance.csv')
    importances.to_csv(importance_path, index=False)
    print(f"\nFeature Importance CSV: {importance_path}")

# ============================================================
# PLOTS
# ============================================================

# Plot 1: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = df_comparison['Model']
test_rmse = df_comparison['Test_RMSE']
test_r2 = df_comparison['Test_R2']

axes[0].bar(models, test_rmse, alpha=0.8, color='steelblue')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('RMSE (kWh)')
axes[0].set_title('Test RMSE - 15min Forecast (Tages-Features)')
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(models, test_r2, alpha=0.8, color='green')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('R²')
axes[1].set_title('Test R² - 15min Forecast (Tages-Features)')
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'model_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 2: Forecast Timeline (Last 3 Days)
fig, ax = plt.subplots(figsize=(16, 6))

end_time = y_test.index[-1]
start_time = end_time - pd.Timedelta(days=3)
mask = (y_test.index >= start_time) & (y_test.index <= end_time)

ax.plot(y_test.index[mask], y_test.values[mask], label='True', linewidth=1.5)
ax.plot(y_test.index[mask], best['y_test_pred'][mask], label='Predicted (15min ahead)', 
        linewidth=1.5, linestyle='--', alpha=0.8)
ax.set_title(f'15-min Forecast (Last 3 Days) - {best["name"]}')
ax.set_xlabel('Time')
ax.set_ylabel('Consumption (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_timeline.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 3: Scatter
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(y_test.values, best['y_test_pred'], alpha=0.3, s=1)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('True (kWh)')
ax.set_ylabel('Predicted (kWh)')
ax.set_title(f'Prediction Quality - {best["name"]}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'scatter.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 4: Feature Importance
if hasattr(best['model'], 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(importances)), importances['importance'])
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {best["name"]}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'feature_importance.png'), dpi=Config.PLOT_DPI)
    plt.close()

# Plot 5: Residuals
fig, ax = plt.subplots(figsize=(12, 6))
residuals = y_test.values - best['y_test_pred']
ax.scatter(y_test.values, residuals, alpha=0.3, s=1)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('True Values (kWh)')
ax.set_ylabel('Residuals (kWh)')
ax.set_title(f'Residuals - {best["name"]}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'residuals.png'), dpi=Config.PLOT_DPI)
plt.close()

print(f"\nPlots gespeichert: {Config.OUTPUT_DIR}")

# ============================================================
# SAVE
# ============================================================

dump(best['model'], os.path.join(Config.OUTPUT_DIR, 'model_15min.joblib'))
np.save(os.path.join(Config.OUTPUT_DIR, 'features_15min.npy'), X_train.columns.to_numpy())
df_comparison.to_csv(os.path.join(Config.OUTPUT_DIR, 'model_comparison.csv'), index=False)

config_data = {
    'model_name': best['name'],
    'test_rmse': float(best['test_rmse']),
    'test_mae': float(best['test_mae']),
    'test_r2': float(best['test_r2']),
    'overfitting': float(best['overfitting']),
    'features': feature_cols,
    'params': best['params'],
    'forecast_horizon': Config.FORECAST_HORIZON_NAME,
    'trained_date': datetime.now().isoformat(),
}

with open(os.path.join(Config.OUTPUT_DIR, 'config_15min.json'), 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\nModel gespeichert: {Config.OUTPUT_DIR}")

print("\n" + "=" * 70)
print("V17 FERTIG - 15-MIN FORECAST MIT TAGES-FEATURES")
print("=" * 70)
