"""
OPTIMIERTES FORECASTING-FRAMEWORK
==================================
Features:
- Regularisierung gegen Overfitting
- Time Series Cross-Validation
- Feature Selection & Importance
- Experiment Logging
- Multiple Forecast Horizonte
- Baseline Models zum Vergleich
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import json
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Pfade
    ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
    DATA_DIR = os.path.join(REPO_ROOT, "Data")
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "modeling_results")
    
    # Zeitraum
    START_DATE = "2021-01-01 00:00:00"
    END_DATE = "2024-12-31 23:45:00"
    
    # Train/Test Split (zeitlich)
    TRAIN_RATIO = 0.8
    
    # Cross-Validation
    CV_SPLITS = 5  # Time Series Split
    
    # Feature Selection
    USE_FEATURE_SELECTION = True
    MAX_FEATURES = 30  # Top N wichtigste Features
    
    # Regularisierung (gegen Overfitting)
    RF_PARAMS = {
        'n_estimators': 100,  # Weniger Trees = weniger Overfitting
        'max_depth': 15,      # WICHTIG: Tiefe begrenzen!
        'min_samples_split': 20,  # Mehr Samples nötig für Split
        'min_samples_leaf': 10,   # Mehr Samples in Leaf
        'max_features': 'sqrt',   # Nur sqrt(n) Features pro Split
        'random_state': 42,
        'n_jobs': -1,
    }
    
    GB_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.05,  # Kleinere Learning Rate
        'max_depth': 5,         # WICHTIG: Flachere Trees!
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'subsample': 0.8,       # Nur 80% der Daten pro Tree
        'random_state': 42,
    }
    
    # Feature Engineering
    TIME_FEATURES = ['hour', 'dayofweek', 'month', 'dayofyear']
    
    # Plots
    PLOT_DPI = 150

# Verzeichnisse erstellen
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("🚀 OPTIMIERTES FORECASTING-FRAMEWORK")
print("=" * 70)

# ============================================================
# 1) DATEN LADEN & VORBEREITEN
# ============================================================

print("\n[1/7] Lade und bereite Daten vor...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    parse_dates=["DateTime"],
    index_col="DateTime",
).sort_index()

# Zeitfenster
df = df.loc[Config.START_DATE:Config.END_DATE].copy()
print(f" Zeitraum: {df.index.min()} → {df.index.max()} ({len(df)} Zeilen)")

# ============================================================
# 2) FEATURE ENGINEERING
# ============================================================

print("\n[2/7] Feature Engineering...")

target_col = "Stromverbrauch"
y = df[target_col].astype(float).copy()

X_all = df.drop(columns=[target_col]).copy()

# Zeitfeatures hinzufügen
for feat in Config.TIME_FEATURES:
    if feat == 'hour':
        X_all[feat] = X_all.index.hour
    elif feat == 'dayofweek':
        X_all[feat] = X_all.index.dayofweek
    elif feat == 'month':
        X_all[feat] = X_all.index.month
    elif feat == 'dayofyear':
        X_all[feat] = X_all.index.dayofyear

# Object-Spalten entfernen
obj_cols = X_all.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    print(f"  ⚠ Entferne {len(obj_cols)} Object-Spalten: {list(obj_cols)}")
    X_all = X_all.drop(columns=obj_cols)

# ⚠️ KRITISCH: Entferne Features, die zu viel Leakage verursachen
# Diff_15min ist direkt abgeleitet von Lags → kann zu Overfitting führen
suspicious_features = ['Diff_15min']
for feat in suspicious_features:
    if feat in X_all.columns:
        print(f"  ⚠ Entferne verdächtiges Feature: {feat}")
        X_all = X_all.drop(columns=[feat])

# Alles zu float konvertieren
X_all = X_all.astype(float)

# NaNs bereinigen
mask = y.notna()
X = X_all.loc[mask].dropna(axis=0)
y = y.loc[X.index]

print(f" Feature-Matrix: {X.shape}")
print(f" Features: {X.shape[1]}")

# ============================================================
# 3) TRAIN/TEST SPLIT (Zeitlich!)
# ============================================================

print("\n[3/7] Train/Test Split (zeitlich)...")

n_train = int(len(X) * Config.TRAIN_RATIO)

X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

print(f" Train: {X_train.index.min()} → {X_train.index.max()} ({len(X_train)} Zeilen)")
print(f" Test : {X_test.index.min()} → {X_test.index.max()} ({len(X_test)} Zeilen)")

# ============================================================
# 4) BASELINE MODEL (Naive Forecast)
# ============================================================

print("\n[4/7] Berechne Baseline (Naive Forecast: Lag_15min)...")

# Naive Forecast: Einfach den letzten Wert nehmen
if 'Lag_15min' in X_test.columns:
    y_test_naive = X_test['Lag_15min']
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_test_naive))
    baseline_mae = mean_absolute_error(y_test, y_test_naive)
    baseline_r2 = r2_score(y_test, y_test_naive)
    
    print(f" Baseline RMSE: {baseline_rmse:.2f}")
    print(f" Baseline MAE : {baseline_mae:.2f}")
    print(f" Baseline R²  : {baseline_r2:.3f}")
else:
    print("  ⚠ Lag_15min nicht gefunden, überspringe Baseline")
    baseline_rmse = None

# ============================================================
# 5) FEATURE SELECTION (Optional)
# ============================================================

print("\n[5/7] Feature Selection...")

if Config.USE_FEATURE_SELECTION and X_train.shape[1] > Config.MAX_FEATURES:
    print(f"  → Verwende Feature Selection (Top {Config.MAX_FEATURES})")
    
    # Quick Random Forest für Feature Importance
    rf_quick = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_quick.fit(X_train, y_train)
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_quick.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top N Features
    top_features = feature_importance.head(Config.MAX_FEATURES)['feature'].tolist()
    
    print(f"   Ausgewählte Features ({len(top_features)}):")
    for i, (feat, imp) in enumerate(zip(top_features[:10], 
                                         feature_importance.head(10)['importance']), 1):
        print(f"    {i:2d}. {feat:40s} {imp:.4f}")
    
    # Feature Selection anwenden
    X_train = X_train[top_features]
    X_test = X_test[top_features]
    
    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = feature_importance.head(20)
    ax.barh(range(len(top_20)), top_20['importance'])
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Feature Importance (Quick RF)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'feature_importance_quick.png'), 
                dpi=Config.PLOT_DPI)
    plt.close()
    print(f"   Plot gespeichert: feature_importance_quick.png")
else:
    print(f"  → Keine Feature Selection (verwende alle {X_train.shape[1]} Features)")
    top_features = X_train.columns.tolist()

# ============================================================
# 6) MODEL TRAINING & EVALUATION
# ============================================================

print("\n[6/7] Trainiere Modelle...")

results = []

def train_and_evaluate(model, name, params):
    """Trainiert und evaluiert ein Modell mit Cross-Validation"""
    print(f"\n  === {name} ===")
    
    # Haupt-Training
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metriken
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  TRAIN:")
    print(f"    RMSE: {train_rmse:.2f}")
    print(f"    MAE : {train_mae:.2f}")
    print(f"    R²  : {train_r2:.4f}")
    print(f"  TEST:")
    print(f"    RMSE: {test_rmse:.2f}")
    print(f"    MAE : {test_mae:.2f}")
    print(f"    R²  : {test_r2:.4f}")
    
    # Overfitting Check
    overfitting_score = train_r2 - test_r2
    if overfitting_score > 0.1:
        print(f"  ⚠️  OVERFITTING DETECTED: ΔR² = {overfitting_score:.4f}")
    else:
        print(f"   Overfitting under control: ΔR² = {overfitting_score:.4f}")
    
    # Cross-Validation (Time Series Split)
    print(f"  → Cross-Validation ({Config.CV_SPLITS} splits)...")
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Neues Modell für jede Fold
        model_cv = model.__class__(**params)
        model_cv.fit(X_tr, y_tr)
        y_val_pred = model_cv.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        fold_r2 = r2_score(y_val, y_val_pred)
        cv_scores.append({'rmse': fold_rmse, 'r2': fold_r2})
        
        print(f"    Fold {fold}: RMSE={fold_rmse:.2f}, R²={fold_r2:.4f}")
    
    cv_rmse_mean = np.mean([s['rmse'] for s in cv_scores])
    cv_rmse_std = np.std([s['rmse'] for s in cv_scores])
    cv_r2_mean = np.mean([s['r2'] for s in cv_scores])
    
    print(f"   CV RMSE: {cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}")
    print(f"   CV R²  : {cv_r2_mean:.4f}")
    
    # Ergebnisse speichern
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
        'overfitting_score': overfitting_score,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'cv_r2_mean': cv_r2_mean,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
    }
    
    return result

# ============================================================
# MODELLE TRAINIEREN
# ============================================================

# 1) Random Forest (regularisiert)
rf_model = RandomForestRegressor(**Config.RF_PARAMS)
results.append(train_and_evaluate(rf_model, "RandomForest (regularized)", Config.RF_PARAMS))

# 2) Gradient Boosting (regularisiert)
gb_model = GradientBoostingRegressor(**Config.GB_PARAMS)
results.append(train_and_evaluate(gb_model, "GradientBoosting (regularized)", Config.GB_PARAMS))

# 3) Ridge Regression (als Baseline)
ridge_params = {'alpha': 10.0, 'random_state': 42}
ridge_model = Ridge(**ridge_params)
results.append(train_and_evaluate(ridge_model, "Ridge Regression", ridge_params))

# ============================================================
# 7) ERGEBNISSE & VISUALISIERUNGEN
# ============================================================

print("\n[7/7] Erstelle Visualisierungen & Reports...")

# Übersichtstabelle
print("\n  === MODELLVERGLEICH ===")
comparison = []
for r in results:
    comparison.append({
        'Model': r['name'],
        'Test_RMSE': r['test_rmse'],
        'Test_MAE': r['test_mae'],
        'Test_R2': r['test_r2'],
        'CV_RMSE': r['cv_rmse_mean'],
        'Overfitting': r['overfitting_score']
    })

df_comparison = pd.DataFrame(comparison).sort_values('Test_RMSE')
print(df_comparison.to_string(index=False))

# Bestes Modell
best_model_result = min(results, key=lambda x: x['test_rmse'])
best_name = best_model_result['name']
print(f"\n  ✅ BESTES MODELL: {best_name}")
print(f"     Test RMSE: {best_model_result['test_rmse']:.2f}")
print(f"     Test R²  : {best_model_result['test_r2']:.4f}")

# Baseline-Vergleich
if baseline_rmse:
    improvement = (baseline_rmse - best_model_result['test_rmse']) / baseline_rmse * 100
    print(f"\n  📊 VERBESSERUNG vs. BASELINE:")
    print(f"     Baseline RMSE: {baseline_rmse:.2f}")
    print(f"     Best Model RMSE: {best_model_result['test_rmse']:.2f}")
    print(f"     Improvement: {improvement:.1f}%")

# Plot 1: Forecast vs. True (letzte 7 Tage)
fig, ax = plt.subplots(figsize=(16, 6))
last_week = best_model_result['y_test'].last('7D')
pred_last_week = pd.Series(
    best_model_result['y_test_pred'], 
    index=best_model_result['y_test'].index
).last('7D')

ax.plot(last_week.index, last_week.values, label='True', linewidth=1.5, alpha=0.8)
ax.plot(pred_last_week.index, pred_last_week.values, label='Predicted', 
        linewidth=1.5, alpha=0.8, linestyle='--')
ax.set_title(f'Forecast vs. True (letzte 7 Tage) - {best_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'forecast_last_7days.png'), dpi=Config.PLOT_DPI)
plt.close()
print("   Plot: forecast_last_7days.png")

# Plot 2: Residuals
fig, ax = plt.subplots(figsize=(12, 6))
residuals = best_model_result['y_test'] - best_model_result['y_test_pred']
ax.scatter(best_model_result['y_test'], residuals, alpha=0.3, s=1)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('True Values (kWh)')
ax.set_ylabel('Residuals (kWh)')
ax.set_title(f'Residuals Plot - {best_name}', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, 'residuals.png'), dpi=Config.PLOT_DPI)
plt.close()
print(" Plot: residuals.png")

# Plot 3: Feature Importance (bestes Modell)
if hasattr(best_model_result['model'], 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 8))
    importances = best_model_result['model'].feature_importances_
    feature_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    ax.barh(range(len(feature_imp)), feature_imp['importance'])
    ax.set_yticks(range(len(feature_imp)))
    ax.set_yticklabels(feature_imp['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top 20 Features - {best_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'feature_importance_final.png'), 
                dpi=Config.PLOT_DPI)
    plt.close()
    print("   Plot: feature_importance_final.png")

# ============================================================
# MODELL & METADATA SPEICHERN
# ============================================================

# Bestes Modell speichern
model_path = os.path.join(Config.OUTPUT_DIR, 'best_model.joblib')
dump(best_model_result['model'], model_path)
print(f"\n   Modell gespeichert: {model_path}")

# Features speichern
features_path = os.path.join(Config.OUTPUT_DIR, 'features.npy')
np.save(features_path, X_train.columns.to_numpy())
print(f"   Features gespeichert: {features_path}")

# Experiment Log
experiment_log = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'train_ratio': Config.TRAIN_RATIO,
        'cv_splits': Config.CV_SPLITS,
        'feature_selection': Config.USE_FEATURE_SELECTION,
        'max_features': Config.MAX_FEATURES if Config.USE_FEATURE_SELECTION else 'all',
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
        'mae': float(baseline_mae) if baseline_rmse else None,
        'r2': float(baseline_r2) if baseline_rmse else None,
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
        'overfitting_score': float(r['overfitting_score']),
        'cv_rmse_mean': float(r['cv_rmse_mean']),
        'cv_rmse_std': float(r['cv_rmse_std']),
    })

experiment_log['best_model'] = best_name

# Log speichern
log_path = os.path.join(Config.OUTPUT_DIR, 'experiment_log.json')
with open(log_path, 'w') as f:
    json.dump(experiment_log, f, indent=2)
print(f"   Experiment Log gespeichert: {log_path}")


print(" MODELLIERUNG ABGESCHLOSSEN!")

