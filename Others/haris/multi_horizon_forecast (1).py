"""
MULTI-HORIZON FORECASTING
=========================
Testet verschiedene Forecast-Horizonte:
- 15min (trivial, Baseline)
- 1h (realistisch)
- 4h (interessant)
- 24h (schwierig)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "multi_horizon_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(" MULTI-HORIZON FORECASTING TEST")
print("=" * 70)

# ============================================================
# DATEN LADEN
# ============================================================

print("\n[1/3] Lade Daten...")

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    encoding="latin1",
    parse_dates=["DateTime"],
    index_col="DateTime",
).sort_index()

df = df.loc["2021-01-01":"2024-12-31"].copy()
print(f" Geladen: {len(df)} Zeilen")

# ============================================================
# MULTI-HORIZON FORECASTING FUNCTION
# ============================================================

def forecast_horizon(df, horizon_steps, horizon_name):
    """
    Trainiert ein Modell für einen bestimmten Forecast-Horizont
    
    Args:
        df: DataFrame mit allen Daten
        horizon_steps: Anzahl 15-min Steps voraus (z.B. 4 = 1 Stunde)
        horizon_name: Name für Output (z.B. "1h")
    """
    print(f"\n{'='*70}")
    print(f"  HORIZON: {horizon_name} ({horizon_steps} steps = {horizon_steps*15}min)")
    print(f"{'='*70}")
    
    # Target: X Steps in die Zukunft
    target_col = "Stromverbrauch"
    y = df[target_col].shift(-horizon_steps).astype(float)
    
    # Features
    X_all = df.drop(columns=[target_col]).copy()
    
    # Zeit-Features
    X_all['hour'] = X_all.index.hour
    X_all['dayofweek'] = X_all.index.dayofweek
    X_all['month'] = X_all.index.month
    X_all['dayofyear'] = X_all.index.dayofyear
    
    # Object-Spalten entfernen
    obj_cols = X_all.select_dtypes(include="object").columns
    X_all = X_all.drop(columns=obj_cols, errors='ignore')
    X_all = X_all.astype(float)
    
    # NaNs bereinigen (wichtig wegen shift!)
    valid_idx = y.notna() & X_all.notna().all(axis=1)
    X = X_all.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()
    
    print(f"  Daten nach Shift: {len(X)} Zeilen (verloren: {len(df) - len(X)})")
    
    # Train/Test Split
    n_train = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    
    # ============================================================
    # WICHTIG: Feature Selection basierend auf Horizont
    # ============================================================
    
    # Für kurze Horizonte: Lags sind Gold
    # Für lange Horizonte: Zeit + Wetter wichtiger
    
    if horizon_steps <= 4:  # ≤1h
        # Kurz: Verwende alle Lags
        lag_features = [c for c in X_train.columns if 'Lag_' in c]
        time_features = ['hour', 'dayofweek', 'month', 'dayofyear']
        keep_features = lag_features + time_features
    else:  # >1h
        # Lang: Nur Lag_24h + Zeit + Wetter
        # (Lag_15min hilft nicht mehr bei 4h forecast!)
        lag_features = [c for c in X_train.columns if 'Lag_24h' in c]
        time_features = ['hour', 'dayofweek', 'month', 'dayofyear']
        weather_features = [c for c in X_train.columns if '_lag15' in c][:10]  # Top 10 Wetter
        keep_features = lag_features + time_features + weather_features
    
    # Features filtern
    keep_features = [f for f in keep_features if f in X_train.columns]
    X_train = X_train[keep_features]
    X_test = X_test[keep_features]
    
    print(f"  Features: {len(keep_features)}")
    print(f"  Top Features: {keep_features[:5]}")
    
    # ============================================================
    # MODEL TRAINING
    # ============================================================
    
    # Regularisierung abhängig vom Horizont
    if horizon_steps <= 4:
        # Kurz: Weniger Regularisierung (einfaches Problem)
        max_depth = 20
        min_samples_leaf = 5
    else:
        # Lang: Mehr Regularisierung (schwieriges Problem)
        max_depth = 10
        min_samples_leaf = 20
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"  Training mit max_depth={max_depth}, min_samples_leaf={min_samples_leaf}...")
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
    
    # Naive Baseline: Einfach Lag_15min nehmen (falls vorhanden)
    if 'Lag_15min' in df.columns:
        # Für Horizont X: Nehme Wert von vor X steps
        baseline_col = f'Lag_{horizon_steps*15}min' if f'Lag_{horizon_steps*15}min' in df.columns else 'Lag_24h'
        if baseline_col in X_test.columns:
            y_baseline = X_test[baseline_col]
            baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
            baseline_r2 = r2_score(y_test, y_baseline)
            improvement = (baseline_rmse - test_rmse) / baseline_rmse * 100
        else:
            baseline_rmse = None
            improvement = None
    else:
        baseline_rmse = None
        improvement = None
    
    # ============================================================
    # ERGEBNISSE
    # ============================================================
    
    print(f"\n  RESULTS:")
    print(f"  {'TRAIN:':<20} RMSE={train_rmse:>8.2f}  MAE={train_mae:>8.2f}  R²={train_r2:>7.4f}")
    print(f"  {'TEST:':<20} RMSE={test_rmse:>8.2f}  MAE={test_mae:>8.2f}  R²={test_r2:>7.4f}")
    
    if baseline_rmse:
        print(f"  {'BASELINE:':<20} RMSE={baseline_rmse:>8.2f}  R²={baseline_r2:>7.4f}")
        print(f"  {'IMPROVEMENT:':<20} {improvement:>6.1f}%")
    
    overfitting = train_r2 - test_r2
    if overfitting > 0.05:
        print(f"    OVERFITTING: ΔR² = {overfitting:.4f}")
    else:
        print(f"   Overfitting OK: ΔR² = {overfitting:.4f}")
    
    # ============================================================
    # PLOT: Forecast vs. True (letzte 7 Tage)
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Letzte 7 Tage
    plot_data = pd.DataFrame({
        'true': y_test,
        'pred': y_test_pred
    })
    end_time = plot_data.index[-1]
    start_time = end_time - pd.Timedelta(days=7)
    plot_data = plot_data.loc[start_time:end_time]
    
    ax.plot(plot_data.index, plot_data['true'], label='True', linewidth=1.5, alpha=0.8)
    ax.plot(plot_data.index, plot_data['pred'], label='Predicted', 
            linewidth=1.5, alpha=0.8, linestyle='--')
    
    ax.set_title(f'Forecast {horizon_name} ahead - Last 7 Days', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stromverbrauch (kWh)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Metriken in Plot
    textstr = f'Test RMSE: {test_rmse:.0f}\nTest R²: {test_r2:.4f}'
    if baseline_rmse:
        textstr += f'\nImprovement: {improvement:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'forecast_{horizon_name}.png'), dpi=150)
    plt.close()
    
    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        ax.barh(range(len(importances)), importances['importance'])
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(importances['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - Horizon {horizon_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'importance_{horizon_name}.png'), dpi=150)
        plt.close()
    
    return {
        'horizon': horizon_name,
        'horizon_steps': horizon_steps,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting': overfitting,
        'baseline_rmse': baseline_rmse,
        'improvement': improvement,
        'n_features': len(keep_features)
    }

# ============================================================
# TESTE VERSCHIEDENE HORIZONTE
# ============================================================

print("\n[2/3] Teste verschiedene Forecast-Horizonte...")

horizons = [
    (1, "15min"),
    (4, "1h"),
    (16, "4h"),
    (96, "24h"),
]

results = []

for steps, name in horizons:
    result = forecast_horizon(df, steps, name)
    results.append(result)

# ============================================================
# ZUSAMMENFASSUNG
# ============================================================

print("\n[3/3] Zusammenfassung")

print("\n" + "=" * 70)
print("  MULTI-HORIZON COMPARISON")
print("=" * 70)

comparison = pd.DataFrame(results)
comparison = comparison[['horizon', 'test_rmse', 'test_r2', 'overfitting', 
                          'baseline_rmse', 'improvement', 'n_features']]

print(comparison.to_string(index=False))

# Vergleichs-Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RMSE
ax = axes[0, 0]
horizons_names = comparison['horizon']
ax.bar(horizons_names, comparison['test_rmse'], alpha=0.7, color='steelblue')
if comparison['baseline_rmse'].notna().any():
    ax.scatter(horizons_names, comparison['baseline_rmse'], 
               color='red', s=100, marker='D', label='Baseline', zorder=5)
ax.set_ylabel('RMSE (kWh)')
ax.set_title('Test RMSE by Horizon', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: R²
ax = axes[0, 1]
ax.bar(horizons_names, comparison['test_r2'], alpha=0.7, color='green')
ax.set_ylabel('R²')
ax.set_title('Test R² by Horizon', fontweight='bold')
ax.axhline(0.95, color='red', linestyle='--', linewidth=1, label='R²=0.95')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Overfitting
ax = axes[1, 0]
ax.bar(horizons_names, comparison['overfitting'], alpha=0.7, color='orange')
ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Threshold=0.05')
ax.set_ylabel('ΔR² (Train - Test)')
ax.set_title('Overfitting Score', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Improvement vs. Baseline
ax = axes[1, 1]
valid_improvement = comparison[comparison['improvement'].notna()]
if len(valid_improvement) > 0:
    ax.bar(valid_improvement['horizon'], valid_improvement['improvement'], 
           alpha=0.7, color='purple')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('ML Improvement vs. Naive Baseline', fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'horizon_comparison.png'), dpi=150)
plt.close()

print("\n Vergleichs-Plot gespeichert: horizon_comparison.png")

print(f" FERTIG! Alle Ergebnisse in: {OUTPUT_DIR}")
