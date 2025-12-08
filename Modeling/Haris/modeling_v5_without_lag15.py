"""
MODELING FRAMEWORK V5 - ALL FEATURES WITHOUT LAG_15MIN
=======================================================
Alle Features (inkl. Wetter), aber OHNE Lag_15min
Modelle: RandomForest, GradientBoosting, Ridge, XGBoost
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
    OUTPUT_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V5"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
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
    
    RIDGE_PARAMS = {
        'alpha': 10.0,
        'random_state': 42,
    }
    
    PLOT_DPI = 150

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("MODELING FRAMEWORK V5 - ALL FEATURES WITHOUT LAG_15MIN")
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
# FEATURE SELECTION - ALLE FEATURES OHNE LAG_15MIN
# ============================================================

print("\n[2/5] Feature Selection (Ohne Lag_15min)...")

target_col = "Stromverbrauch"
y = df[target_col].astype(float).copy()

X_all = df.drop(columns=[target_col]).copy()

# Object-Spalten entfernen
obj_cols = X_all.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    print(f"Entferne {len(obj_cols)} Object-Spalten: {list(obj_cols)}")
    X_all = X_all.drop(columns=obj_cols)

# LAG_15MIN ENTFERNEN
if 'Lag_15min' in X_all.columns:
    print(f"\nEntferne Lag_15min (zu dominant)")
    X_all = X_all.drop(columns=['Lag_15min'])

X_all = X_all.astype(float)

# NaNs bereinigen
mask = y.notna()
X = X_all.loc[mask].dropna(axis=0)
y = y.loc[X.index]

print(f"\nFinale Feature-Matrix: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Feature-Kategorien zählen
lag_features = [c for c in X.columns if 'Lag_' in c]
weather_features = [c for c in X.columns if '_lag15' in c]
time_features = [c for c in X.columns if c in ['Monat', 'Wochentag', 'Quartal', 'Woche des Jahres', 'Tag des Jahres']]
other_features = [c for c in X.columns if c not in lag_features + weather_features + time_features]

print(f"\nFeature-Kategorien:")
print(f"  Stromverbrauch-Lags: {len(lag_features)} (ohne Lag_15min)")
print(f"  Wetter-Features: {len(weather_features)}")
print(f"  Zeit-Features: {len(time_features)}")
print(f"  Sonstige: {len(other_features)}")

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

# Baseline mit Lag_30min (da Lag_15min entfernt wurde)
if 'Lag_30min' in X_test.columns:
    y_test_naive = X_test['Lag_30min']
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_test_naive))
    baseline_mae = mean_absolute_error(y_test, y_test_naive)
    baseline_r2 = r2_score(y_test, y_test_naive)
    print(f"Baseline (Lag_30min) RMSE: {baseline_rmse:.2f}")
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
    
    # Overfitting Check
    overfitting = train_r2 - test_r2
    print(f"  Overfitting (Train R² - Test R²): {overfitting:.4f}")
    
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
        'overfitting': overfitting,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'cv_r2_mean': cv_r2_mean,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
    }
    
    return result

# Train models
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
# VERGLEICH
# ============================================================

print("\n" + "=" * 70)
print("VERGLEICH - ALLE FEATURES OHNE LAG_15MIN")
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

# Plot 1: Model Comparison - RMSE
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = df_comparison['Model']
test_rmse = df_comparison['Test_RMSE']
test_r2 = df_comparison['Test_R2']

axes[0].bar(models, test_rmse, alpha=0.8, color='steelblue')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('RMSE (kWh)')
axes[0].set_title('Model Comparison - Test RMSE')
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(models, test_r2, alpha=0.8, color='green')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('R²')
axes[1].set_title('Model Comparison - Test R²')
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'model_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 2: Overfitting Comparison
fig, ax = plt.subplots(figsize=(10, 6))
overfitting_scores = df_comparison['Overfitting']

colors = ['red' if x > 0.05 else 'green' for x in overfitting_scores]
ax.bar(models, overfitting_scores, alpha=0.8, color=colors)
ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Threshold 0.05')
ax.set_xlabel('Model')
ax.set_ylabel('Overfitting Score (Train R² - Test R²)')
ax.set_title('Overfitting Comparison')
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'overfitting_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 3: Forecast vs. True
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
ax.set_title(f'Forecast vs. True (Without Lag_15min) - {best_name}')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 4: Residuals
fig, ax = plt.subplots(figsize=(12, 6))
residuals = best_model_result['y_test'] - best_model_result['y_test_pred']
ax.scatter(best_model_result['y_test'], residuals, alpha=0.3, s=1)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('True Values (kWh)')
ax.set_ylabel('Residuals (kWh)')
ax.set_title(f'Residuals (Without Lag_15min) - {best_name}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'residuals.png'), dpi=Config.PLOT_DPI)
plt.close()

# Plot 5: Feature Importance (Best Model)
if hasattr(best_model_result['model'], 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model_result['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    n_plot = min(30, len(importances))
    importances_plot = importances.head(n_plot)
    
    ax.barh(range(len(importances_plot)), importances_plot['importance'])
    ax.set_yticks(range(len(importances_plot)))
    ax.set_yticklabels(importances_plot['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance (Top {n_plot}) - {best_name}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'feature_importance.png'), dpi=Config.PLOT_DPI)
    plt.close()
    
    # Save feature importance to CSV
    importance_path = os.path.join(Config.OUTPUT_DIR, 'feature_importance.csv')
    importances.to_csv(importance_path, index=False)
    print(f"Feature Importance gespeichert: {importance_path}")

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

# Feature list
features_list_path = os.path.join(Config.OUTPUT_DIR, 'features_used.txt')
with open(features_list_path, 'w') as f:
    f.write("FEATURES USED IN V5 (WITHOUT LAG_15MIN)\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total Features: {len(X.columns)}\n\n")
    f.write(f"Stromverbrauch-Lags ({len(lag_features)}):\n")
    for feat in lag_features:
        f.write(f"  - {feat}\n")
    f.write(f"\nWetter-Features ({len(weather_features)}):\n")
    for feat in weather_features[:10]:
        f.write(f"  - {feat}\n")
    if len(weather_features) > 10:
        f.write(f"  ... und {len(weather_features)-10} weitere\n")
    f.write(f"\nZeit-Features ({len(time_features)}):\n")
    for feat in time_features:
        f.write(f"  - {feat}\n")
    f.write(f"\nSonstige ({len(other_features)}):\n")
    for feat in other_features:
        f.write(f"  - {feat}\n")

experiment_log = {
    'timestamp': datetime.now().isoformat(),
    'version': 'V5',
    'description': 'All Features (incl. Weather) WITHOUT Lag_15min',
    'config': {
        'train_ratio': Config.TRAIN_RATIO,
        'cv_splits': Config.CV_SPLITS,
        'feature_selection': 'All features except Lag_15min',
        'removed_features': ['Lag_15min'],
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
        'method': 'Lag_30min',
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
        'overfitting': float(r['overfitting']),
        'cv_rmse_mean': float(r['cv_rmse_mean']),
        'cv_rmse_std': float(r['cv_rmse_std']),
    })

experiment_log['best_model'] = best_name

log_path = os.path.join(Config.OUTPUT_DIR, 'experiment_log.json')
with open(log_path, 'w') as f:
    json.dump(experiment_log, f, indent=2)

print(f"\nModell gespeichert: {model_path}")
print(f"Features gespeichert: {features_path}")
print(f"Feature Liste gespeichert: {features_list_path}")
print(f"Comparison gespeichert: {comparison_path}")
print(f"Log gespeichert: {log_path}")

print("\n" + "=" * 70)
print("V5 FERTIG - ALLE FEATURES OHNE LAG_15MIN")
print("=" * 70)
