
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
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

class Config:
    BASE_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V10_Production"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "models")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    for dir in [OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR]:
        os.makedirs(dir, exist_ok=True)
    
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
    
    START_DATE = "2015-01-01 00:00:00"
    END_DATE = "2024-12-31 23:45:00"
    
    FORECAST_HORIZON_STEPS = 4
    FORECAST_HORIZON_NAME = "1h"
    
    TRAIN_RATIO = 0.7
    CV_SPLITS = 5
    
    SELECTED_FEATURES = [
        'Lag_24h',
        'Diffusstrahlung; Zehnminutenmittel_lag15',
        'Globalstrahlung; Zehnminutenmittel_lag15',
        'Sonnenscheindauer; Zehnminutensumme_lag15',
        'Lufttemperatur 2 m u. Boden_lag15',
        'relative Luftfeuchtigkeit_lag15',
        'Lufttemperatur 2 m u. Gras_lag15',
        'Chilltemperatur_lag15',
        'Boenspitze (3-Sekundenboe); Maximum in km/h_lag15',
        'Boenspitze (Sekundenboe)_lag15',
        'Boenspitze (3-Sekundenboe); Maximum in m/s_lag15',
        'Lufttemperatur Bodenoberflaeche_lag15',
        'IstSonntag',
        'IstArbeitstag',
    ]
    
    RF_PARAMS = {
        'n_estimators': 500,
        'max_depth': 25,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
    }
    
    GB_PARAMS = {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 7,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'subsample': 0.9,
        'random_state': 42,
    }
    
    XGB_PARAMS = {
        'n_estimators': 250,
        'learning_rate': 0.03,
        'max_depth': 4,
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

print("=" * 80)
print("V10 PRODUCTION - 1H FORECAST MIT ZYKLISCHEN FEATURES")
print("=" * 80)

print("\n[1/6] Lade Daten...")

df = pd.read_csv(
    Config.INPUT_FILE,
    sep=";",
    encoding="latin1",
    index_col=0,
    parse_dates=True
).sort_index()

df = df.loc[Config.START_DATE:Config.END_DATE].copy()

if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

print(f"Zeitraum: {df.index.min()} bis {df.index.max()}")
print(f"Zeilen: {len(df):,}")

print("\n[2/6] Erstelle zyklische Features...")

df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)

df['weekday_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

print("Zyklische Features erstellt:")
print("  - hour_sin, hour_cos (Stunde)")
print("  - month_sin, month_cos (Monat)")
print("  - weekday_sin, weekday_cos (Wochentag)")

print("\n[3/6] Target & Features...")

target_col = "Stromverbrauch"
y = df[target_col].shift(-Config.FORECAST_HORIZON_STEPS).astype(float)

X_all = df.drop(columns=[target_col]).copy()

obj_cols = X_all.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    X_all = X_all.drop(columns=obj_cols)

X_all = X_all.astype(float)

cyclic_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
all_features = Config.SELECTED_FEATURES + cyclic_features

available_features = [f for f in all_features if f in X_all.columns]
missing_features = [f for f in all_features if f not in X_all.columns]

if missing_features:
    print(f"\nFehlende Features ({len(missing_features)}):")
    for f in missing_features[:5]:
        print(f"  - {f}")

X_all = X_all[available_features]

print(f"\nFeatures gesamt: {len(available_features)}")
print(f"  Wetter: {len([f for f in available_features if 'lag15' in f])}")
print(f"  Zyklisch: {len(cyclic_features)}")
print(f"  Sonstige: {len([f for f in available_features if f not in cyclic_features and 'lag15' not in f])}")

mask = y.notna() & X_all.notna().all(axis=1)
X = X_all.loc[mask].copy()
y = y.loc[mask].copy()

print(f"\nSamples: {len(X):,}")

print("\n[4/6] Train/Test Split...")

n_train = int(len(X) * Config.TRAIN_RATIO)

X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

print(f"Train: {len(X_train):,} ({Config.TRAIN_RATIO*100:.0f}%)")
print(f"Test:  {len(X_test):,} ({(1-Config.TRAIN_RATIO)*100:.0f}%)")

print("\n[5/6] Training...")

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
    
    print(f"  Train: RMSE={train_rmse:.2f}, R²={train_r2:.4f}")
    print(f"  Test:  RMSE={test_rmse:.2f}, R²={test_r2:.4f}")
    print(f"  Overfitting: {(train_r2 - test_r2):.4f}")
    
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        m = model.__class__(**params)
        m.fit(X_tr, y_tr)
        cv_scores.append(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))
    
    cv_rmse = np.mean(cv_scores)
    print(f"  CV: RMSE={cv_rmse:.2f} ± {np.std(cv_scores):.2f}")
    
    return {
        'name': name,
        'model': model,
        'params': params,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'train_r2': train_r2,
        'overfitting': train_r2 - test_r2,
        'cv_rmse': cv_rmse,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
    }

rf = RandomForestRegressor(**Config.RF_PARAMS)
results.append(train_and_evaluate(rf, "RandomForest", Config.RF_PARAMS))

gb = GradientBoostingRegressor(**Config.GB_PARAMS)
results.append(train_and_evaluate(gb, "GradientBoosting", Config.GB_PARAMS))

if HAS_XGB:
    xgb_model = xgb.XGBRegressor(**Config.XGB_PARAMS)
    results.append(train_and_evaluate(xgb_model, "XGBoost", Config.XGB_PARAMS))

ridge = Ridge(**Config.RIDGE_PARAMS)
results.append(train_and_evaluate(ridge, "Ridge", Config.RIDGE_PARAMS))

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

comparison = pd.DataFrame([{
    'Model': r['name'],
    'Test_RMSE': r['test_rmse'],
    'Test_R2': r['test_r2'],
    'Overfitting': r['overfitting'],
    'CV_RMSE': r['cv_rmse'],
} for r in results]).sort_values('Test_RMSE')

print(comparison.to_string(index=False))

best = min(results, key=lambda x: x['test_rmse'])

print(f"\nBEST: {best['name']}")
print(f"RMSE: {best['test_rmse']:.2f} kWh")
print(f"R²:   {best['test_r2']:.4f}")

print("\n[6/6] Feature Importance & Plots...")

if hasattr(best['model'], 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    for i, row in importances.head(10).iterrows():
        print(f"  {row['feature'][:40]:<40s} {row['importance']:.4f}")
    
    importances.to_csv(os.path.join(Config.RESULTS_DIR, 'feature_importance.csv'), index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models = comparison['Model']
test_rmse = comparison['Test_RMSE']
test_r2 = comparison['Test_R2']

axes[0].bar(models, test_rmse, alpha=0.8, color='steelblue')
axes[0].set_ylabel('RMSE (kWh)')
axes[0].set_title('Model Comparison - Test RMSE')
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(models, test_r2, alpha=0.8, color='green')
axes[1].set_ylabel('R²')
axes[1].set_title('Model Comparison - Test R²')
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'model_comparison.png'), dpi=Config.PLOT_DPI)
plt.close()

fig, ax = plt.subplots(figsize=(18, 6))
end_time = y_test.index[-1]
start_time = end_time - pd.Timedelta(days=7)
mask = (y_test.index >= start_time) & (y_test.index <= end_time)

ax.plot(y_test.index[mask], y_test.values[mask], label='Actual', linewidth=1.5)
ax.plot(y_test.index[mask], best['y_test_pred'][mask], label='Predicted', 
        linewidth=1.5, linestyle='--', alpha=0.8)
ax.set_title(f'1h Forecast Last 7 Days - {best["name"]}')
ax.set_xlabel('Time')
ax.set_ylabel('Consumption (kWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, 'forecast_7days.png'), dpi=Config.PLOT_DPI)
plt.close()

if hasattr(best['model'], 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_n = min(15, len(importances))
    top_features = importances.head(top_n)
    
    y_pos = range(len(top_features))
    ax.barh(y_pos, top_features['importance'], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features - {best["name"]}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'feature_importance.png'), dpi=Config.PLOT_DPI)
    plt.close()

dump(best['model'], os.path.join(Config.OUTPUT_DIR, 'v10_model.joblib'))
np.save(os.path.join(Config.OUTPUT_DIR, 'v10_features.npy'), X_train.columns.to_numpy())
comparison.to_csv(os.path.join(Config.RESULTS_DIR, 'model_comparison.csv'), index=False)

config_data = {
    'model_name': best['name'],
    'test_rmse': float(best['test_rmse']),
    'test_r2': float(best['test_r2']),
    'train_r2': float(best['train_r2']),
    'overfitting': float(best['overfitting']),
    'cv_rmse': float(best['cv_rmse']),
    'features': list(X_train.columns),
    'cyclic_features': cyclic_features,
    'params': best['params'],
    'forecast_horizon': Config.FORECAST_HORIZON_NAME,
    'trained_date': datetime.now().isoformat(),
}

with open(os.path.join(Config.OUTPUT_DIR, 'v10_config.json'), 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\nModel: {Config.OUTPUT_DIR}")
print(f"Plots: {Config.PLOTS_DIR}")
print(f"Results: {Config.RESULTS_DIR}")

print("\n" + "=" * 80)
print("V10 PRODUCTION MODELING COMPLETE")
print("=" * 80)
