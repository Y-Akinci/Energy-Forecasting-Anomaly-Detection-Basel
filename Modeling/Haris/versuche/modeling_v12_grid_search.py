"""
MODELING FRAMEWORK V11 - GRID SEARCH HYPERPARAMETER TUNING
===========================================================
1h Forecast mit Grid Search für GradientBoosting und XGBoost
Basierend auf V10 Features (ohne dominante Lags)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V11"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    START_DATE = "2015-01-01 00:00:00"
    END_DATE = "2024-12-31 23:45:00"
    
    # Forecast Horizon
    FORECAST_HORIZON_STEPS = 4
    FORECAST_HORIZON_NAME = "1h"
    
    TRAIN_RATIO = 0.7
    CV_SPLITS = 5
    
    # Features (gleich wie V10)
    SELECTED_FEATURES = [
        'Lag_1h',
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
        'Böenspitze (3-Sekundenböe); Maximum in km/h_lag15',
        'Böenspitze (Sekundenböe)_lag15',
        'Böenspitze (3-Sekundenböe); Maximum in m/s_lag15',
        'Lufttemperatur Bodenoberfläche_lag15',
        'Monat',
        'Wochentag',
        'Quartal',
    ]
    
    # Model Parameters (für RF und Ridge)
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
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
print("MODELING FRAMEWORK V11 - GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 70)

# ============================================================
# DATEN LADEN
# ============================================================

print("\n[1/5] Lade Daten...")

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

print(f"\n[2/5] Target Shift & Feature Selection...")

target_col = "Stromverbrauch"
y = df[target_col].shift(-Config.FORECAST_HORIZON_STEPS).astype(float)

print(f"Target: {Config.FORECAST_HORIZON_NAME} voraus (shift={-Config.FORECAST_HORIZON_STEPS})")

# Features
X_all = df.drop(columns=[target_col]).copy()

# Object-Spalten entfernen
obj_cols = X_all.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    print(f"\nEntferne {len(obj_cols)} Object-Spalten")
    X_all = X_all.drop(columns=obj_cols)

X_all = X_all.astype(float)

# Nur ausgewählte Features
available_features = [f for f in Config.SELECTED_FEATURES if f in X_all.columns]
missing_features = [f for f in Config.SELECTED_FEATURES if f not in X_all.columns]

if missing_features:
    print(f"\nFehlende Features: {len(missing_features)}")
    for feat in missing_features:
        print(f"  - {feat}")

X_all = X_all[available_features]

print(f"\nVerwendete Features: {len(available_features)}")

# NaNs bereinigen
mask = y.notna() & X_all.notna().all(axis=1)
X = X_all.loc[mask].copy()
y = y.loc[mask].copy()

print(f"\nNach Bereinigung:")
print(f"  Features: {X.shape[1]}")
print(f"  Samples: {X.shape[0]}")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

print("\n[3/5] Train/Test Split...")

n_train = int(len(X) * Config.TRAIN_RATIO)

X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")

# ============================================================
# BASELINE
# ============================================================

print("\n[4/5] Baseline...")

if 'Lag_1h' in X_test.columns:
    y_test_naive = X_test['Lag_1h']
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_test_naive))
    baseline_mae = mean_absolute_error(y_test, y_test_naive)
    baseline_r2 = r2_score(y_test, y_test_naive)
    print(f"Baseline (Lag_1h) RMSE: {baseline_rmse:.2f}")
    print(f"Baseline MAE:  {baseline_mae:.2f}")
    print(f"Baseline R²:   {baseline_r2:.4f}")
else:
    baseline_rmse = None
    baseline_mae = None
    baseline_r2 = None

# ============================================================
# MODEL TRAINING WITH GRID SEARCH
# ============================================================

print("\n[5/5] Trainiere Modelle mit Grid Search...")

results = []

def evaluate_model(model, name, params):
    print(f"\n{name} - Evaluation:")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
    
    overfitting = train_r2 - test_r2
    print(f"  Overfitting: {overfitting:.4f}")
    
    # Manual CV for comparison
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if hasattr(model, 'get_params'):
            model_cv = model.__class__(**model.get_params())
        else:
            model_cv = model.__class__(**params)
        model_cv.fit(X_tr, y_tr)
        y_val_pred = model_cv.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        fold_r2 = r2_score(y_val, y_val_pred)
        cv_scores.append({'rmse': fold_rmse, 'r2': fold_r2})
    
    cv_rmse_mean = np.mean([s['rmse'] for s in cv_scores])
    cv_rmse_std = np.std([s['rmse'] for s in cv_scores])
    cv_r2_mean = np.mean([s['r2'] for s in cv_scores])
    
    print(f"  CV    RMSE: {cv_rmse_mean:.2f} +/- {cv_rmse_std:.2f}, R²: {cv_r2_mean:.4f}")
    
    result = {
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
        'cv_rmse_std': cv_rmse_std,
        'cv_r2_mean': cv_r2_mean,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
    }
    
    return result

# RandomForest (Standard)
print("\n" + "="*70)
print("RANDOMFOREST (Standard)")
print("="*70)
rf_model = RandomForestRegressor(**Config.RF_PARAMS)
rf_model.fit(X_train, y_train)
results.append(evaluate_model(rf_model, "RandomForest", Config.RF_PARAMS))

# GradientBoosting mit Grid Search
print("\n" + "="*70)
print("GRADIENTBOOSTING (Grid Search)")
print("="*70)

gb_param_grid = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.03, 0.05, 0.08],
    'max_depth': [5, 6, 7],
    'subsample': [0.8, 0.9],
    'min_samples_split': [20],
    'min_samples_leaf': [10],
    'random_state': [42]
}

n_combinations = (len(gb_param_grid['n_estimators']) * 
                  len(gb_param_grid['learning_rate']) * 
                  len(gb_param_grid['max_depth']) * 
                  len(gb_param_grid['subsample']))

print(f"Teste {n_combinations} Kombinationen...")
print("Das kann mehrere Minuten dauern...")

gb_grid = GridSearchCV(
    estimator=GradientBoostingRegressor(),
    param_grid=gb_param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

gb_grid.fit(X_train, y_train)

print(f"\nBeste Parameter:")
for param, value in gb_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"Bester CV Score (RMSE): {-gb_grid.best_score_:.2f}")

gb_best = gb_grid.best_estimator_
results.append(evaluate_model(gb_best, "GradientBoosting_GridSearch", gb_grid.best_params_))

# XGBoost mit Grid Search
if HAS_XGB:
    print("\n" + "="*70)
    print("XGBOOST (Grid Search)")
    print("="*70)
    
    xgb_param_grid = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.03, 0.05, 0.08],
        'max_depth': [5, 6, 7],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [10],
        'random_state': [42],
        'n_jobs': [-1]
    }
    
    n_combinations = (len(xgb_param_grid['n_estimators']) * 
                      len(xgb_param_grid['learning_rate']) * 
                      len(xgb_param_grid['max_depth']) * 
                      len(xgb_param_grid['subsample']) * 
                      len(xgb_param_grid['colsample_bytree']))
    
    print(f"Teste {n_combinations} Kombinationen...")
    print("Das kann mehrere Minuten dauern...")
    
    xgb_grid = GridSearchCV(
        estimator=xgb.XGBRegressor(),
        param_grid=xgb_param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    
    xgb_grid.fit(X_train, y_train)
    
    print(f"\nBeste Parameter:")
    for param, value in xgb_grid.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Bester CV Score (RMSE): {-xgb_grid.best_score_:.2f}")
    
    xgb_best = xgb_grid.best_estimator_
    results.append(evaluate_model(xgb_best, "XGBoost_GridSearch", xgb_grid.best_params_))

# Ridge (Standard)
print("\n" + "="*70)
print("RIDGE (Standard)")
print("="*70)
ridge_model = Ridge(**Config.RIDGE_PARAMS)
ridge_model.fit(X_train, y_train)
results.append(evaluate_model(ridge_model, "Ridge", Config.RIDGE_PARAMS))

# ============================================================
# VERGLEICH
# ============================================================

print("\n" + "=" * 70)
print("VERGLEICH - GRID SEARCH RESULTS")
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
        'CV_Std': r['cv_rmse_std'],
    })

df_comparison = pd.DataFrame(comparison).sort_values('Test_RMSE')
print(df_comparison.to_string(index=False))

best_model_result = min(results, key=lambda x: x['test_rmse'])
best_name = best_model_result['name']

print(f"\nBestes Modell: {best_name}")
print(f"Test RMSE: {best_model_result['test_rmse']:.2f}")
print(f"Test R²:   {best_model_result['test_r2']:.4f}")
print(f"Overfitting: {best_model_result['overfitting']:.4f}")

if baseline_rmse:
    improvement = (baseline_rmse - best_model_result['test_rmse']) / baseline_rmse * 100
    print(f"\nBaseline RMSE:     {baseline_rmse:.2f}")
    print(f"Best Model RMSE:   {best_model_result['test_rmse']:.2f}")
    print(f"Verbesserung:      {improvement:.1f}%")

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
axes[0].set_title('Test RMSE - Grid Search Results')
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(models, test_r2, alpha=0.8, color='green')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('R²')
axes[1].set_title('Test R² - Grid Search Results')
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'model_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 2: Forecast vs. True
fig, ax = plt.subplots(figsize=(16, 6))

plot_data = pd.DataFrame({
    'true': best_model_result['y_test'],
    'pred': best_model_result['y_test_pred']
})

end_time = plot_data.index[-1]
start_time = end_time - pd.Timedelta(days=7)
plot_data_7d = plot_data.loc[start_time:end_time]

ax.plot(plot_data_7d.index, plot_data_7d['true'], label='True', linewidth=1.5)
ax.plot(plot_data_7d.index, plot_data_7d['pred'], label='Predicted (Grid Search)', 
        linewidth=1.5, linestyle='--')
ax.set_title(f'Forecast vs. True - {best_name}')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 3: Feature Importance
if hasattr(best_model_result['model'], 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
    
    importance_path = os.path.join(Config.OUTPUT_DIR, 'feature_importance.csv')
    importances.to_csv(importance_path, index=False)

print(f"\nPlots gespeichert in: {Config.OUTPUT_DIR}")

# ============================================================
# SPEICHERN
# ============================================================

model_path = os.path.join(Config.OUTPUT_DIR, 'best_model.joblib')
dump(best_model_result['model'], model_path)

features_path = os.path.join(Config.OUTPUT_DIR, 'features.npy')
np.save(features_path, X_train.columns.to_numpy())

comparison_path = os.path.join(Config.OUTPUT_DIR, 'model_comparison.csv')
df_comparison.to_csv(comparison_path, index=False)

# Save Grid Search results
grid_search_results = {
    'timestamp': datetime.now().isoformat(),
    'version': 'V11',
    'description': 'Grid Search Hyperparameter Tuning',
    'config': {
        'train_ratio': Config.TRAIN_RATIO,
        'cv_splits': Config.CV_SPLITS,
        'forecast_horizon': Config.FORECAST_HORIZON_NAME,
    },
    'best_model': best_name,
    'best_params': best_model_result['params'],
    'models': []
}

for r in results:
    grid_search_results['models'].append({
        'name': r['name'],
        'params': r['params'],
        'test_rmse': float(r['test_rmse']),
        'test_r2': float(r['test_r2']),
        'overfitting': float(r['overfitting']),
    })

log_path = os.path.join(Config.OUTPUT_DIR, 'grid_search_results.json')
with open(log_path, 'w') as f:
    json.dump(grid_search_results, f, indent=2)

print(f"\nModell gespeichert: {model_path}")
print(f"Grid Search Results gespeichert: {log_path}")

print("\n" + "=" * 70)
print("V11 FERTIG - GRID SEARCH COMPLETE")
print("=" * 70)
