## Experiment: baseline_full (2020-08-31 – 2024-12-31, Split 70/30)

### Experiment 1
Config:
- USE_WEATHER = True
- USE_LAGS = True
- USE_CALENDAR = True
- EXCLUDE_WEATHER = []
- EXCLUDE_LAGS = []
- Modelle: LinearRegression, XGBoost

Results:
- LinearRegression:
  - MAE_train = 257.382009
  - RMSE_train = 353.311516
  - R2_train   = 0.997682
  - MAE_test   = 286.503165
  - RMSE_test  = 646.709154
  - R2_test    = 0.991473

- XGBoost:
  - MAE_train = 169.635443
  - RMSE_train = 228.945862
  - R2_train   = 0.999027
  - MAE_test   = 214.118272
  - RMSE_test  = 306.088784
  - R2_test    = 0.998090


### Experiment 2
Config:
- USE_WEATHER = FALSE
- USE_LAGS = True
- USE_CALENDAR = True
- EXCLUDE_WEATHER = []
- Modelle: XGBoost

Results:
- XGBoost:
  - MAE_train = 179.023606
  - RMSE_train = 242.772294
  - R2_train   = 0.998906
  - MAE_test   = 217.910317
  - RMSE_test  = 310.494800
  - R2_test    = 0.998034

### Experiment 3
Config:
- USE_WEATHER = FALSE
- USE_LAGS = FALSE
- USE_CALENDAR = True
- EXCLUDE_WEATHER = []
- EXCLUDE_LAGS = []
- Modelle: XGBoost

Results:
- XGBoost:
  - MAE_train = 1141.343888
  - RMSE_train = 1568.054223
  - R2_train   = 0.954349
  - MAE_test   = 1518.315463
  - RMSE_test  = 2071.662321
  - R2_test    = 0.912493

### Experiment 3
Config:
- USE_WEATHER = FALSE
- USE_LAGS = FALSE
- USE_CALENDAR = True
- EXCLUDE_WEATHER = []
- EXCLUDE_LAGS = ["Grundversorgte Kunden_Lag_15min", Freie Kunden_Lag_15min"]
- Modelle: XGBoost

Results:
- XGBoost:
  - MAE_train = 171.106273
  - RMSE_train = 230.811486
  - R2_train   = 0.999011
  - MAE_test   = 212.614254
  - RMSE_test  = 304.720824
  - R2_test    = 0.998107


### Experiment 4
Config:
- USE_WEATHER = FALSE
- USE_LAGS = FALSE
- USE_CALENDAR = True
- EXCLUDE_WEATHER = ["BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in km/h_lag15",
   "BÃ¶enspitze (SekundenbÃ¶e); Maximum in km/h_lag15",
   "Böenspitze (Sekundenböe)_lag15",
   "Luftdruck reduziert auf Meeresniveau (QFF)_lag15",
   "Luftdruck reduziert auf Meeresniveau mit Standardatmosphäre (QNH)_lag15",
   "Lufttemperatur Bodenoberfläche_lag15",
   "Windgeschwindigkeit; Zehnminutenmittel in km/h_lag15"]
- EXCLUDE_LAGS = []
- Modelle: XGBoost

Results:
- XGBoost:
  - MAE_train = 169.762065
  - RMSE_train = 229.400521
  - R2_train   = 0.999023
  - MAE_test   = 214.266919
  - RMSE_test  = 306.638279
  - R2_test    = 0.998083




