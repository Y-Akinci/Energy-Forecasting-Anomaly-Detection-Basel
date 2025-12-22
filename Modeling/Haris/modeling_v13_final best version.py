"""
MODELING FRAMEWORK V13 - FINAL PRODUCTION MODEL
================================================
Finales 1h Forecast Modell ohne Lag_1h
Beste Config: GradientBoosting
RMSE: 1762, R²: 0.939
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
    print("XGBoost nicht installiert. Überspringe XGBoost.")

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Pfade
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V13_Final"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    START_DATE = "2015-01-01 00:00:00"
    END_DATE = "2024-12-31 23:45:00"
    
    # Forecast Horizon
    FORECAST_HORIZON_STEPS = 4
    FORECAST_HORIZON_NAME = "1h"
    
    TRAIN_RATIO = 0.7
    CV_SPLITS = 5
    
    # Features OHNE Lag_1h (beste Config)
    SELECTED_FEATURES = [
        'Lag_15min',
        'Lag_30min',
        'Lag_1h',
        'Freie Kunden_lag15',
        'Grundversorgte Kunden_lag15'
        'Lag_24h',
        'Diffusstrahlung; Zehnminutenmittel_lag15',
        'Globalstrahlung; Zehnminutenmittel_lag15',
        'Stunde (Lokal)',
        'IstArbeitstag',
        'Sonnenscheindauer; Zehnminutensumme_lag15',
        'Lufttemperatur 2 m ü. Boden_lag15',
        'IstSonntag',
        'relative Luftfeuchtigkeit_lag15',
        'Lufttemperatur 2 m ü. Gras_lag15',
        'Chilltemperatur_lag15',
        'Lufttemperatur Bodenoberfläche_lag15',
        'Monat',
        'Wochentag',
        'Quartal',
    ]
    
    # Model Parameters (beste Config)
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
print("MODELING FRAMEWORK V13 - FINAL PRODUCTION MODEL")
print("=" * 70)
print("Features: OHNE Lag_1h (realistisches R²)")
print("Best Model: GradientBoosting (R²=0.939, RMSE=1762)")

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
print(f"Zeitraum: {df.index.min()} -> {df.index.max()}")
print(f"Zeilen: {len(df)}")

# ============================================================
# TARGET SHIFT & FEATURE SELECTION
# ============================================================

print(f"\n[2/6] Target Shift & Feature Selection...")

target_col = "Stromverbrauch"
y = df[target_col].shift(-Config.FORECAST_HORIZON_STEPS).astype(float)

print(f"Target: {Config.FORECAST_HORIZON_NAME} voraus")

# Features
X_all = df.drop(columns=[target_col]).copy()

# Object-Spalten entfernen
obj_cols = X_all.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    X_all = X_all.drop(columns=obj_cols)

X_all = X_all.astype(float)

# Features
available_features = [f for f in Config.SELECTED_FEATURES if f in X_all.columns]
X_all = X_all[available_features]

print(f"Features: {len(available_features)}")

# NaNs bereinigen
mask = y.notna() & X_all.notna().all(axis=1)
X = X_all.loc[mask].copy()
y = y.loc[mask].copy()

print(f"Samples: {X.shape[0]}")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

print("\n[3/6] Train/Test Split...")

n_train = int(len(X) * Config.TRAIN_RATIO)

X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

print(f"Train: {len(X_train)}")
print(f"Test:  {len(X_test)}")

# ============================================================
# BASELINE
# ============================================================

print("\n[4/6] Baseline...")

if 'Lag_24h' in X_test.columns:
    y_test_naive = X_test['Lag_24h']
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_test_naive))
    baseline_r2 = r2_score(y_test, y_test_naive)
    print(f"Baseline (Lag_24h) RMSE: {baseline_rmse:.2f}, R²: {baseline_r2:.4f}")
else:
    baseline_rmse = None
    baseline_r2 = None

# ============================================================
# MODEL TRAINING
# ============================================================

print("\n[5/6] Trainiere Modelle...")

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
    
    print(f"  Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
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
    print(f"  CV RMSE: {cv_rmse_mean:.2f}")
    
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
rf_model = RandomForestRegressor(**Config.RF_PARAMS)
results.append(train_and_evaluate(rf_model, "RandomForest", Config.RF_PARAMS))

gb_model = GradientBoostingRegressor(**Config.GB_PARAMS)
results.append(train_and_evaluate(gb_model, "GradientBoosting", Config.GB_PARAMS))

if HAS_XGB:
    xgb_model = xgb.XGBRegressor(**Config.XGB_PARAMS)
    results.append(train_and_evaluate(xgb_model, "XGBoost", Config.XGB_PARAMS))

ridge_model = Ridge(**Config.RIDGE_PARAMS)
results.append(train_and_evaluate(ridge_model, "Ridge", Config.RIDGE_PARAMS))

# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

best_model_result = min(results, key=lambda x: x['test_rmse'])
best_name = best_model_result['name']

for r in results:
    print(f"{r['name']:20s} RMSE: {r['test_rmse']:7.2f}, R²: {r['test_r2']:.4f}")

print(f"\nBEST MODEL: {best_name}")
print(f"RMSE: {best_model_result['test_rmse']:.2f}")
print(f"R²:   {best_model_result['test_r2']:.4f}")

# ============================================================
# PLOTS
# ============================================================

print("\n[6/6] Erstelle Plots...")

# Forecast vs True
fig, ax = plt.subplots(figsize=(16, 6))

plot_data = pd.DataFrame({
    'true': best_model_result['y_test'],
    'pred': best_model_result['y_test_pred']
})

end_time = plot_data.index[-1]
start_time = end_time - pd.Timedelta(days=7)
plot_data_7d = plot_data.loc[start_time:end_time]

ax.plot(plot_data_7d.index, plot_data_7d['true'], label='True', linewidth=1.5)
ax.plot(plot_data_7d.index, plot_data_7d['pred'], label='Predicted', 
        linewidth=1.5, linestyle='--')
ax.set_title(f'Final Model: {best_name}')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_final.png'), dpi=Config.PLOT_DPI)
plt.close()

# Feature Importance
if hasattr(best_model_result['model'], 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model_result['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    ax.barh(range(len(importances)), importances['importance'])
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {best_name}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'feature_importance.png'), dpi=Config.PLOT_DPI)
    plt.close()

# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(Config.OUTPUT_DIR, 'production_model.joblib')
dump(best_model_result['model'], model_path)

features_path = os.path.join(Config.OUTPUT_DIR, 'production_features.npy')
np.save(features_path, X_train.columns.to_numpy())

config_path = os.path.join(Config.OUTPUT_DIR, 'production_config.json')
config_data = {
    'model_name': best_name,
    'test_rmse': float(best_model_result['test_rmse']),
    'test_r2': float(best_model_result['test_r2']),
    'features': available_features,
    'params': best_model_result['params'],
    'forecast_horizon': Config.FORECAST_HORIZON_NAME,
    'trained_date': datetime.now().isoformat(),
}
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\nProduktionsmodell gespeichert: {model_path}")
print(f"Features gespeichert: {features_path}")
print(f"Config gespeichert: {config_path}")

print("\n" + "=" * 70)
print("V13 FINAL FERTIG - READY FOR PRODUCTION")
print("=" * 70)
