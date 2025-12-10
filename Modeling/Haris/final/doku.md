================================================================================
FINAL MODELING FRAMEWORK - PRODUCTION VERSION
================================================================================

[1/7] Lade Daten...
Zeitraum: 2015-01-01 00:00:00+00:00 bis 2024-12-31 23:45:00+00:00
Zeilen: 350,648

[2/7] Target & Features...

Fehlende Features (6):
  - Lufttemperatur 2 m u. Boden_lag15
  - Lufttemperatur 2 m u. Gras_lag15
  - Boenspitze (3-Sekundenboe); Maximum in km/h_lag15
  - Boenspitze (Sekundenboe)_lag15
  - Boenspitze (3-Sekundenboe); Maximum in m/s_lag15

Features vor OneHotEncoding: 15
Samples: 350,647

[3/7] OneHotEncoding kategoriale Features...
Kategoriale Features: 4
Nach OneHotEncoding: 47
Finale Features: 58

[4/7] Train/Test Split...
Train: 245,452 samples (70%)
Test:  105,195 samples (30%)

[5/7] Baseline...
Baseline (Lag_15min): RMSE=1090.03 kWh, R²=0.9767

[6/7] Training Modelle...

RandomForest:
  Train: RMSE=427.12, R²=0.9976
  Test:  RMSE=566.76, R²=0.9937
  Overfitting: 0.0039
  CV: RMSE=614.08 ± 66.08

GradientBoosting:
  Train: RMSE=390.98, R²=0.9980
  Test:  RMSE=478.65, R²=0.9955
  Overfitting: 0.0025
  CV: RMSE=446.39 ± 33.16

XGBoost:
  Train: RMSE=527.44, R²=0.9964
  Test:  RMSE=566.95, R²=0.9937
  Overfitting: 0.0027
  CV: RMSE=548.38 ± 33.25

Ridge:
  Train: RMSE=569.87, R²=0.9958
  Test:  RMSE=583.35, R²=0.9933
  Overfitting: 0.0024
  CV: RMSE=567.65 ± 12.83

================================================================================
RESULTS
================================================================================
           Model  Test_RMSE   Test_MAE  Test_R2  Overfitting    CV_RMSE
GradientBoosting 478.650922 347.134455 0.995500     0.002501 446.387565
    RandomForest 566.759051 429.018111 0.993690     0.003924 614.082580
         XGBoost 566.951525 429.706601 0.993686     0.002675 548.380367
           Ridge 583.351974 441.938584 0.993315     0.002437 567.648749

BEST MODEL: GradientBoosting
Test RMSE: 478.65 kWh
Test R²:   0.9955

Baseline: 1090.03 kWh
Verbesserung: 56.1%

[7/7] Feature Importance & Plots...

Top 15 Features:
  Lag_15min                                          0.9822
  Lag_1h                                             0.0073
  Stunde (Lokal)_6.0                                 0.0018
  Lag_24h                                            0.0016
  Globalstrahlung; Zehnminutenmittel_lag15           0.0013
  Stunde (Lokal)_5.0                                 0.0011
  Stunde (Lokal)_7.0                                 0.0010
  Stunde (Lokal)_20.0                                0.0005
  IstArbeitstag                                      0.0004
  Diffusstrahlung; Zehnminutenmittel_lag15           0.0003
  Stunde (Lokal)_23.0                                0.0003
  Stunde (Lokal)_2.0                                 0.0003
  Stunde (Lokal)_0.0                                 0.0002
  Stunde (Lokal)_19.0                                0.0002
  Stunde (Lokal)_8.0                                 0.0001

Model: C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\Final\models\final_model.joblib
Features: C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\Final\models\final_features.npy
Config: C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\Final\models\final_config.json
Results: C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\Final\results
Plots: C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\Final\plots

================================================================================
MODELING COMPLETE
================================================================================
PS C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model> 