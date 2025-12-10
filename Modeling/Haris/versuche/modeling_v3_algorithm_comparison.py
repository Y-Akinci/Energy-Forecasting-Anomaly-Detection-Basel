"""
MODELING FRAMEWORK V3 - ALGORITHM COMPARISON
=============================================
Testet 7 verschiedene Algorithmen
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump
import json
from datetime import datetime

# Boosting Libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost nicht installiert. Überspringe XGBoost.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM nicht installiert. Überspringe LightGBM.")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost nicht installiert. Überspringe CatBoost.")

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
    DATA_DIR = os.path.join(REPO_ROOT, "Data")
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "modeling_results_v3")
    
    START_DATE = "2021-01-01 00:00:00"
    END_DATE = "2024-12-31 23:45:00"
    
    TRAIN_RATIO = 0.8
    CV_SPLITS = 5
    
    # Model Parameters
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
    
    LGB_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    CATBOOST_PARAMS = {
        'iterations': 100,
        'learning_rate': 0.05,
        'depth': 5,
        'random_state': 42,
        'verbose': False,
    }
    
    RIDGE_PARAMS = {
        'alpha': 10.0,
        'random_state': 42,
    }
    
    ELASTICNET_PARAMS = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'random_state': 42,
    }
    
    SVR_PARAMS = {
        'kernel': 'rbf',
        'C': 100,
        'epsilon': 0.1,
    }
    
    PLOT_DPI = 150

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("MODELING FRAMEWORK V3 - ALGORITHM COMPARISON")
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
# FEATURES & TARGET
# ============================================================

print("\n[2/5] Bereite Features vor...")

target_col = "Stromverbrauch"
y = df[target_col].astype(float).copy()

X_all = df.drop(columns=[target_col]).copy()

# Object-Spalten entfernen
obj_cols = X_all.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    print(f"Entferne {len(obj_cols)} Object-Spalten")
    X_all = X_all.drop(columns=obj_cols)

X_all = X_all.astype(float)

# NaNs bereinigen
mask = y.notna()
X = X_all.loc[mask].dropna(axis=0)
y = y.loc[X.index]

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

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

if 'Lag_15min' in X_test.columns:
    y_test_naive = X_test['Lag_15min']
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_test_naive))
    baseline_mae = mean_absolute_error(y_test, y_test_naive)
    baseline_r2 = r2_score(y_test, y_test_naive)
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"Baseline MAE:  {baseline_mae:.2f}")
    print(f"Baseline R²:   {baseline_r2:.4f}")
else:
    baseline_rmse = None
    baseline_mae = None
    baseline_r2 = None

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
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
    
    # Cross-Validation
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
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
        'cv_rmse_mean': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'cv_r2_mean': cv_r2_mean,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
    }
    
    return result

# Train all models
print("\n=== Training Models ===")

# 1. RandomForest
rf_model = RandomForestRegressor(**Config.RF_PARAMS)
results.append(train_and_evaluate(rf_model, "RandomForest", Config.RF_PARAMS))

# 2. GradientBoosting
gb_model = GradientBoostingRegressor(**Config.GB_PARAMS)
results.append(train_and_evaluate(gb_model, "GradientBoosting", Config.GB_PARAMS))

# 3. XGBoost
if HAS_XGB:
    xgb_model = xgb.XGBRegressor(**Config.XGB_PARAMS)
    results.append(train_and_evaluate(xgb_model, "XGBoost", Config.XGB_PARAMS))

# 4. LightGBM
if HAS_LGB:
    lgb_model = lgb.LGBMRegressor(**Config.LGB_PARAMS)
    results.append(train_and_evaluate(lgb_model, "LightGBM", Config.LGB_PARAMS))

# 5. CatBoost
if HAS_CATBOOST:
    catboost_model = CatBoostRegressor(**Config.CATBOOST_PARAMS)
    results.append(train_and_evaluate(catboost_model, "CatBoost", Config.CATBOOST_PARAMS))

# 6. Ridge
ridge_model = Ridge(**Config.RIDGE_PARAMS)
results.append(train_and_evaluate(ridge_model, "Ridge", Config.RIDGE_PARAMS))

# 7. ElasticNet
elasticnet_model = ElasticNet(**Config.ELASTICNET_PARAMS)
results.append(train_and_evaluate(elasticnet_model, "ElasticNet", Config.ELASTICNET_PARAMS))

# ============================================================
# VERGLEICH
# ============================================================

print("\n" + "=" * 70)
print("ALGORITHM COMPARISON")
print("=" * 70)

comparison = []
for r in results:
    comparison.append({
        'Model': r['name'],
        'Test_RMSE': r['test_rmse'],
        'Test_MAE': r['test_mae'],
        'Test_R2': r['test_r2'],
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

if baseline_rmse:
    improvement = (baseline_rmse - best_model_result['test_rmse']) / baseline_rmse * 100
    print(f"\nBaseline RMSE:     {baseline_rmse:.2f}")
    print(f"Best Model RMSE:   {best_model_result['test_rmse']:.2f}")
    print(f"Verbesserung:      {improvement:.1f}%")

# ============================================================
# PLOTS
# ============================================================

# Plot 1: Model Comparison
fig, ax = plt.subplots(figsize=(12, 6))
models = df_comparison['Model']
test_rmse = df_comparison['Test_RMSE']
cv_rmse = df_comparison['CV_RMSE']

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
ax.bar(x + width/2, cv_rmse, width, label='CV RMSE', alpha=0.8)

ax.set_xlabel('Model')
ax.set_ylabel('RMSE (kWh)')
ax.set_title('Model Comparison - RMSE')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'model_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 2: R² Comparison
fig, ax = plt.subplots(figsize=(12, 6))
r2_scores = df_comparison['Test_R2']

ax.bar(models, r2_scores, alpha=0.8, color='green')
ax.set_xlabel('Model')
ax.set_ylabel('R²')
ax.set_title('Model Comparison - R²')
ax.set_xticklabels(models, rotation=45, ha='right')
ax.axhline(0.99, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'r2_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 3: Forecast vs. True (Best Model)
fig, ax = plt.subplots(figsize=(16, 6))

plot_data = pd.DataFrame({
    'true': best_model_result['y_test'],
    'pred': best_model_result['y_test_pred']
})

end_time = plot_data.index[-1]
start_time = end_time - pd.Timedelta(days=7)
plot_data_7d = plot_data.loc[start_time:end_time]

ax.plot(plot_data_7d.index, plot_data_7d['true'], label='True', linewidth=1.5)
ax.plot(plot_data_7d.index, plot_data_7d['pred'], label='Predicted', linewidth=1.5, linestyle='--')
ax.set_title(f'Forecast vs. True - {best_name}')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_best_model.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 4: Residuals (Best Model)
fig, ax = plt.subplots(figsize=(12, 6))
residuals = best_model_result['y_test'] - best_model_result['y_test_pred']
ax.scatter(best_model_result['y_test'], residuals, alpha=0.3, s=1)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('True Values (kWh)')
ax.set_ylabel('Residuals (kWh)')
ax.set_title(f'Residuals - {best_name}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'residuals_best_model.png'), dpi=Config.PLOT_DPI)
plt.close()

print(f"\nPlots gespeichert in: {Config.OUTPUT_DIR}")

# ============================================================
# SPEICHERN
# ============================================================

model_path = os.path.join(Config.OUTPUT_DIR, 'best_model.joblib')
dump(best_model_result['model'], model_path)

features_path = os.path.join(Config.OUTPUT_DIR, 'features.npy')
np.save(features_path, X_train.columns.to_numpy())

# Comparison CSV
comparison_path = os.path.join(Config.OUTPUT_DIR, 'model_comparison.csv')
df_comparison.to_csv(comparison_path, index=False)

experiment_log = {
    'timestamp': datetime.now().isoformat(),
    'version': 'V3',
    'description': 'Algorithm Comparison: 7 Models',
    'config': {
        'train_ratio': Config.TRAIN_RATIO,
        'cv_splits': Config.CV_SPLITS,
    },
    'data': {
        'train_start': str(X_train.index.min()),
        'train_end': str(X_train.index.max()),
        'test_start': str(X_test.index.min()),
        'test_end': str(X_test.index.max()),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(X_train.columns),
    },
    'baseline': {
        'rmse': float(baseline_rmse) if baseline_rmse else None,
        'mae': float(baseline_mae) if baseline_mae else None,
        'r2': float(baseline_r2) if baseline_r2 else None,
    },
    'models': []
}

for r in results:
    experiment_log['models'].append({
        'name': r['name'],
        'params': r['params'],
        'train_rmse': float(r['train_rmse']),
        'test_rmse': float(r['test_rmse']),
        'train_r2': float(r['train_r2']),
        'test_r2': float(r['test_r2']),
        'cv_rmse_mean': float(r['cv_rmse_mean']),
        'cv_rmse_std': float(r['cv_rmse_std']),
    })

experiment_log['best_model'] = best_name

log_path = os.path.join(Config.OUTPUT_DIR, 'experiment_log.json')
with open(log_path, 'w') as f:
    json.dump(experiment_log, f, indent=2)

print(f"\nModell gespeichert: {model_path}")
print(f"Features gespeichert: {features_path}")
print(f"Comparison gespeichert: {comparison_path}")
print(f"Log gespeichert: {log_path}")

print("\n" + "=" * 70)
print("FERTIG - 7 MODELLE VERGLICHEN")
print("=" * 70)
