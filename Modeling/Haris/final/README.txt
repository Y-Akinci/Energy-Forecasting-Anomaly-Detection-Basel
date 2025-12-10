FINAL PRODUCTION SYSTEM - USAGE GUIDE
======================================

OVERVIEW
--------
Drei saubere Python Files für das komplette Energy Forecasting System:

1. modeling_final.py       - Training & Evaluation
2. forecast_future_final.py - Forecasting bis 6. Dez 2024 + Validation
3. anomaly_detection.py     - Anomalie-Erkennung (historisch & zukunftig)


ORDNER-STRUKTUR
---------------
C:/...Final/
├── models/                 (Trainierte Modelle)
│   ├── final_model.joblib
│   ├── final_features.npy
│   ├── final_config.json
│   └── onehot_encoder.joblib
├── plots/                  (Modeling Plots)
│   ├── model_comparison.png
│   ├── forecast_3days.png
│   ├── scatter.png
│   ├── feature_importance.png
│   └── residuals.png
├── results/                (Modeling Results)
│   ├── feature_importance.csv
│   └── model_comparison.csv
├── forecasts/              (Forecast Outputs)
│   ├── forecast_TIMESTAMP.csv
│   └── summary_TIMESTAMP.json
├── forecast_plots/         (Forecast Plots)
│   ├── forecast_full.png
│   ├── validation_analysis.png
│   ├── hourly_pattern.png
│   └── daily_consumption.png
├── anomalies/              (Anomaly Results)
│   ├── historical_anomalies_TIMESTAMP.csv
│   ├── forecast_anomalies_TIMESTAMP.csv
│   └── anomaly_summary_TIMESTAMP.json
└── anomaly_plots/          (Anomaly Plots)
    ├── anomalies_timeline.png
    ├── anomaly_analysis.png
    ├── anomalies_last30days.png
    └── forecast_anomalies.png


WORKFLOW
--------

1. MODELING (modeling_final.py)
   - Ladt Daten von 2015-2024
   - OneHotEncoding fur kategoriale Features
   - Trainiert 4 Modelle (RF, GB, XGB, Ridge)
   - Cross-Validation mit TimeSeriesSplit
   - Feature Importance Analyse
   - Speichert bestes Modell
   
   Output:
   - Trainiertes Modell
   - Feature Liste
   - Config JSON
   - 5 Plots
   - Model Comparison CSV
   - Feature Importance CSV

2. FORECASTING (forecast_future_final.py)
   - Ladt trainiertes Modell
   - Erstellt Rolling Forecast bis 6. Dez 2024
   - Validierung gegen zum_forecaste_vergleichen.csv
   - Berechnet Metriken (RMSE, MAE, R², MAPE)
   - Visualisierung
   
   Output:
   - Forecast CSV mit Predictions + Actuals
   - Summary JSON
   - 4 Validation Plots
   - Hourly Pattern
   - Daily Consumption

3. ANOMALY DETECTION (anomaly_detection.py)
   - Historische Daten Analyse
   - 3 Methoden: Z-Score, IQR, Isolation Forest
   - Kombinierte Anomalie-Erkennung
   - Future Forecast Anomalien (falls vorhanden)
   - Residual Anomalien
   
   Output:
   - Historical Anomalies CSV
   - Forecast Anomalies CSV
   - Summary JSON
   - 4 Analysis Plots


FEATURES
--------

OPTIMIERUNGEN:
- OneHotEncoding fur Monat, Wochentag, Quartal, Stunde
- Optimierte Hyperparameter (aus Grid Search)
- Lag Features: 15min, 30min, 1h, 24h
- Wetter Features (15min lag)
- Zeit Features

FEATURE IMPORTANCE:
- Top Features werden angezeigt
- CSV Export fur detaillierte Analyse
- Visualisierung in Plots

VALIDATION:
- Test Set (30% der Daten)
- Cross-Validation (5-Fold TimeSeriesSplit)
- Echte Daten Validierung (6. Dez 2024)

METRIKEN:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
- Overfitting Score


EXECUTION ORDER
---------------

python modeling_final.py
   ↓
python forecast_future_final.py
   ↓
python anomaly_detection.py


ERWARTETE RESULTS
-----------------

MODELING:
- Best Model: GradientBoosting oder XGBoost
- Test R²: 0.94-0.97
- Test RMSE: 1500-2000 kWh
- Overfitting: < 0.03

FORECAST:
- Predictions: mehrere tausend (15-min Intervalle)
- Validation R²: 0.90-0.95
- Validation RMSE: 1500-2500 kWh
- MAPE: 3-5%

ANOMALY:
- Historical: 0.5-2% der Daten
- Forecast: wenige Anomalien (hoffentlich!)
- Methods: Z-Score am sensitivsten, IQR am robustesten


WICHTIGE HINWEISE
-----------------

1. KATEGORIALE FEATURES:
   OneHotEncoding wird automatisch angewendet
   Encoder wird gespeichert fur Forecast

2. ROLLING FORECAST:
   Nutzt eigene Predictions als Lags
   Seasonal Wetter-Durchschnitte

3. VALIDATION CSV:
   Muss Spalte "Start der Messung" und "Stromverbrauch" haben
   Format: 2024-12-06T23:45:00+01:00

4. ANOMALIEN:
   Kombinierte Methode = OR (Z-Score OR IQR OR Isolation Forest)
   False Positives moglich bei echten Spitzen


TROUBLESHOOTING
---------------

Problem: "OneHotEncoder not found"
Losung: modeling_final.py muss zuerst laufen

Problem: "Feature mismatch"
Losung: Prüfe ob features in Daten vorhanden sind

Problem: "Validation CSV nicht gefunden"
Losung: Pfad in Config anpassen

Problem: "Memory Error"
Losung: Forecast in kleineren Chunks (reduziere Tage)


CUSTOMIZATION
-------------

Hyperparameter andern:
- Config.RF_PARAMS, GB_PARAMS, XGB_PARAMS in modeling_final.py

Forecast Zeitraum andern:
- Config.FORECAST_TARGET_DATE in forecast_future_final.py

Anomaly Sensitivity andern:
- Config.ZSCORE_THRESHOLD (Standard: 3.0)
- Config.IQR_MULTIPLIER (Standard: 1.5)
- Config.CONTAMINATION (Standard: 0.01)


VERSION INFO
------------
Python: 3.11+
Required Packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost (optional)
- joblib

Created: 2024-12-10
Author: Energy Forecasting System
