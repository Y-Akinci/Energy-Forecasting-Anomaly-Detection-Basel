# MODEL VERSIONS DOCUMENTATION
================================

## V1: Modelling_forecast v1.py
**Datei:** Modelling_forecast v1.py

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00
- Ende: 2024-12-31 23:45:00
- Dauer: 4 Jahre

### 2) FEATURES & ZIEL
- Target: Stromverbrauch
- Features: Alle aus data prep
- Zusätzliche Features: hour, dayofweek, month, dayofyear
- Object-Spalten: Entfernt

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (0.8)
- Methode: Zeitlich (chronologisch)
- Keine Cross-Validation

### 4) MODELLE
**RandomForest:**
- n_estimators: 200
- max_depth: None (unbegrenzt)
- random_state: 42
- n_jobs: -1

**GradientBoosting:**
- n_estimators: 200
- learning_rate: 0.05
- max_depth: 3
- random_state: 42

### 5) METRIKEN
RandomForest:

Train RMSE: 108.16

Test RMSE: 326.54

Train MAE: 76.90

Test MAE: 223.83

Train R²: 1.000

Test R²: 0.998

GradientBoosting:

Train RMSE: 324.81

Test RMSE: 350.67

Train MAE: 236.66

Test MAE: 248.12

Train R²: 0.998

Test R²: 0.997

### 6) ERGEBNISSE
- Bestes Modell:
- Verbesserung vs Baseline:

### 7) BESONDERHEITEN
- Keine Regularisierung bei RandomForest (max_depth=None)
- 4 Zeit-Features manuell hinzugefügt
- Keine Cross-Validation

---

## V2: [Name]
## V2: Modeling Framework mit CV
**Datei:** `C:\...\modeling_framework_v2.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00
- Ende: 2024-12-31 23:45:00
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen im ausgewählten Zeitraum)

### 2) FEATURES & ZIEL
- Target: Stromverbrauch
- Features: Alle numerischen Features aus `processed_merged_features.csv` (39 Features nach Entfernen von Object-Spalten)
- Zusätzliche Features: Enthält u. a. Lag-Features wie `Lag_15min` (für Baseline genutzt)
- Entfernte Features: 2 Object-Spalten entfernt, alle verbleibenden Spalten zu Float gecastet

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112192 Samples, Test: 28048 Samples)
- Methode: Zeitlich (chronologischer Split)
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set, Auswertung über RMSE und R² pro Fold

### 4) MODELLE

**RandomForest**
- n_estimators: 100
- max_depth: 15
- min_samples_split: 20
- min_samples_leaf: 10
- max_features: "sqrt"
- random_state: 42
- n_jobs: -1

**GradientBoosting**
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 5
- min_samples_split: 20
- min_samples_leaf: 10
- subsample: 0.8
- random_state: 42

**Ridge Regression**
- alpha: 10.0
- random_state: 42

### 5) METRIKEN

**Baseline (Naive, Lag_15min auf Test-Set)**
- Test RMSE: 576.46
- Test MAE: 441.33
- Test R²: 0.9927

**RandomForest**
- Train RMSE: 259.36
- Test RMSE: 343.71
- Train MAE: 187.68
- Test MAE: 243.91
- Train R²: 0.9987
- Test R²: 0.9974
- CV RMSE (Train-Set, 5 Folds): 393.66 ± 90.20
- CV R² (Mean): 0.9970

**GradientBoosting**
- Train RMSE: 314.03
- Test RMSE: 339.46
- Train MAE: 229.89
- Test MAE: 239.38
- Train R²: 0.9981
- Test R²: 0.9975
- CV RMSE (Train-Set, 5 Folds): 330.05 ± 16.36
- CV R² (Mean): 0.9979

**Ridge**
- Train RMSE: 369.55
- Test RMSE: 973.20
- Train MAE: 269.16
- Test MAE: 324.42
- Train R²: 0.9974
- Test R²: 0.9793
- CV RMSE (Train-Set, 5 Folds): 394.79 ± 18.99
- CV R² (Mean): 0.9971

### 6) ERGEBNISSE
- Bestes Modell: GradientBoosting (niedrigster Test-RMSE: 339.46, geringster Test-MAE und bestes Test-R²)
- Verbesserung vs Baseline: Test-RMSE von 576.46 (Baseline) auf 339.46 (GradientBoosting), ca. 41.1 % Reduktion des RMSE auf dem Test-Set

### 7) ÄNDERUNGEN ZU V1
- Was wurde geändert:
  - Stärkere Regularisierung der Baummodelle (RandomForest mit begrenzter Tiefe und Mindest-Samples pro Split/Leaf, GradientBoosting mit begrenzter Tiefe und Subsample < 1).
  - Einführung von TimeSeriesSplit-Cross-Validation (5 Folds) auf dem Trainingsdatensatz.
  - Hinzufügen eines linearen Basismodells (Ridge Regression) als zusätzlicher Vergleich.
  - Implementierung einer expliziten Baseline (naiver Lag_15min Forecast) und automatischer Vergleichs-Logik inkl. JSON-Log, Modell- und Feature-Speicherung sowie Plot-Generierung (Forecast vs. True, Residuen).

- Warum:
  - Regularisierung, um Overfitting der Baummodelle aus V1 zu reduzieren und realistischere Testmetriken zu erhalten.
  - Cross-Validation, um robustere Schätzungen der Modellgüte zu bekommen und die Stabilität über verschiedene Zeitfenster im Training zu prüfen.
  - Zusätzliche Baseline und Logging, um die Modelle besser interpretieren, vergleichen und reproduzieren zu können.

- Erwartung:
  - Vergleichbare oder leicht bessere Testmetriken im Verhältnis zur Baseline, aber mit weniger Overfitting und stabileren CV-Werten.
  - Besseres Verständnis, welches Modell über verschiedene Zeitfenster hinweg konsistent performt (GradientBoosting zeigt stabilere CV-RMSE bei gleichzeitig bestem Test-RMSE).



## V3: Algorithm Comparison (7 Modelle)
**Datei:** `C:\...\modeling_v3_algorithm_comparison.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00
- Ende: 2024-12-31 23:45:00
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen im Zeitraum)

### 2) FEATURES & ZIEL
- Target: Stromverbrauch
- Features: Alle numerischen Features aus `processed_merged_features.csv` (39 Features nach Entfernen von 2 Object-Spalten)
- Zusätzliche Features: inklusive Lag-Features wie `Lag_15min` (für Baseline genutzt)
- Entfernte Features: 2 Object-Spalten entfernt, alle verbleibenden Features zu Float gecastet

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112192 Samples, Test: 28048 Samples)
- Methode: Zeitlich (chronologischer Split)
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set, Metriken RMSE und R² pro Fold

### 4) MODELLE

**RandomForest**
- n_estimators: 100
- max_depth: 15
- min_samples_split: 20
- min_samples_leaf: 10
- max_features: "sqrt"
- random_state: 42
- n_jobs: -1

**GradientBoosting**
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 5
- min_samples_split: 20
- min_samples_leaf: 10
- subsample: 0.8
- random_state: 42

**XGBoost**
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 5
- min_child_weight: 10
- subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42
- n_jobs: -1

**LightGBM**
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 5
- min_child_samples: 20
- subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42
- n_jobs: -1
- verbose: -1

**CatBoost**
- iterations: 100
- learning_rate: 0.05
- depth: 5
- random_state: 42
- verbose: False

**Ridge**
- alpha: 10.0
- random_state: 42

**ElasticNet**
- alpha: 1.0
- l1_ratio: 0.5
- random_state: 42

### 5) METRIKEN

**Baseline (Naive, Lag_15min auf Test-Set)**
- Test RMSE: 576.46
- Test MAE: 441.33
- Test R²: 0.9927

**RandomForest**
- Train RMSE: 259.36
- Test RMSE: 343.71
- Train MAE: 187.68
- Test MAE: 243.91
- Train R²: 0.9987
- Test R²: 0.9974
- CV RMSE (Train-Set, 5 Folds): 393.66 ± 90.20
- CV R² (Mean): 0.9970

**GradientBoosting**
- Train RMSE: 314.03
- Test RMSE: 339.46
- Train MAE: 229.89
- Test MAE: 239.38
- Train R²: 0.9981
- Test R²: 0.9975
- CV RMSE (Train-Set, 5 Folds): 330.05 ± 16.36
- CV R² (Mean): 0.9979

**XGBoost**
- Train RMSE: 319.79
- Test RMSE: 342.63
- Train MAE: 234.77
- Test MAE: 243.35
- Train R²: 0.9981
- Test R²: 0.9974
- CV RMSE (Train-Set, 5 Folds): 336.81 ± 18.84
- CV R² (Mean): 0.9978

**LightGBM**
- Train RMSE: 319.77
- Test RMSE: 343.12
- Train MAE: 234.27
- Test MAE: 243.03
- Train R²: 0.9981
- Test R²: 0.9974
- CV RMSE (Train-Set, 5 Folds): 336.87 ± 19.53
- CV R² (Mean): 0.9978

**CatBoost**
- Train RMSE: 390.25
- Test RMSE: 396.24
- Train MAE: 292.26
- Test MAE: 291.54
- Train R²: 0.9971
- Test R²: 0.9966
- CV RMSE (Train-Set, 5 Folds): 420.91 ± 33.60
- CV R² (Mean): 0.9966

**Ridge**
- Train RMSE: 369.55
- Test RMSE: 973.20
- Train MAE: 269.16
- Test MAE: 324.42
- Train R²: 0.9974
- Test R²: 0.9793
- CV RMSE (Train-Set, 5 Folds): 394.79 ± 18.99
- CV R² (Mean): 0.9971

**ElasticNet**
- Train RMSE: 370.65
- Test RMSE: 391.53
- Train MAE: 269.84
- Test MAE: 275.87
- Train R²: 0.9974
- Test R²: 0.9966
- CV RMSE (Train-Set, 5 Folds): 383.07 ± 15.91
- CV R² (Mean): 0.9972

### 6) ERGEBNISSE
- Bestes Modell: GradientBoosting (niedrigster Test-RMSE 339.46, geringster Test-MAE 239.38, bestes Test-R² 0.9975)
- Verbesserung vs Baseline: Test-RMSE von 576.46 (Baseline) auf 339.46 (GradientBoosting), ca. 41.1 % Reduktion des RMSE auf dem Test-Set
- Zusätzliche Outputs:
  - `model_comparison.csv` mit allen Modellen und Metriken
  - Plots: `model_comparison.png` (Test/CV-RMSE), `r2_comparison.png`, `forecast_best_model.png`, `residuals_best_model.png`

### 7) ÄNDERUNGEN ZU V2
- Was wurde geändert:
  - Erweiterung von 3 auf bis zu 7 Modelle (RF, GB, XGBoost, LightGBM, CatBoost, Ridge, ElasticNet).
  - Gemeinsames, generisches Training und Auswerten aller Modelle mit identischer TimeSeriesSplit-CV.
  - Einführung eines expliziten Modellvergleichs (RMSE- und R²-Balkendiagramme, Vergleichs-CSV).
  - Neues Output-Verzeichnis `modeling_results_v3` mit separater Speicherung von Best-Model, Features, Vergleich und Log.

- Warum:
  - Ziel war ein breiter Algorithmus-Vergleich, um zu prüfen, ob komplexere Gradient-Boosting-Varianten (XGBoost, LightGBM, CatBoost) oder lineare Modelle (Ridge, ElasticNet) gegenüber RF/GB Vorteile bringen.
  - Einheitliche Pipeline und CV erlauben einen fairen Vergleich unter gleichen Bedingungen.

- Erwartung:
  - Identifikation eines „global“ besten Modells für diese Datenbasis (hier: GradientBoosting mit sehr guten und stabilen Werten).
  - Besseres Gefühl, ob sich zusätzliche Modellkomplexität wirklich lohnt oder ob klassische GradientBoosting-Modelle bereits ausreichen.

## V4: Feature Selection – No Weather
**Datei:** `C:\...\modeling_v4_no_weather.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00
- Ende: 2024-12-31 23:45:00
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen im Zeitraum) [web:18]

### 2) FEATURES & ZIEL
- Target: Stromverbrauch
- Features: 13 ausgewählte Features (nur Stromverbrauch-Lags, Zeit-Features, einfache Kunden-/Binary-Features)
- Zusätzliche Features:
  - Lag-Features: `Lag_15min`, `Lag_30min`, `Lag_1h`, `Lag_24h`, `Grundversorgte Kunden_Lag_15min`, `Freie Kunden_Lag_15min`
  - Zeit-Features: `Monat`, `Wochentag`, `Quartal`, `Woche des Jahres`, `Tag des Jahres`
  - Binary-Features: `IstSonntag`, `IstArbeitstag`
- Entfernte Features: Alle Wetter-Features explizit ausgeschlossen, 2 Object-Spalten entfernt, alle genutzten Features als Float [web:18]

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112192 Samples, Test: 28048 Samples)
- Methode: Zeitlich (chronologischer Split)
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set (RMSE & R² pro Fold) [web:18]

### 4) MODELLE

**RandomForest**
- n_estimators: 100  
- max_depth: 15  
- min_samples_split: 20  
- min_samples_leaf: 10  
- max_features: "sqrt"  
- random_state: 42  
- n_jobs: -1  

**GradientBoosting**
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_samples_split: 20  
- min_samples_leaf: 10  
- subsample: 0.8  
- random_state: 42  

**XGBoost** (falls installiert)
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_child_weight: 10  
- subsample: 0.8  
- colsample_bytree: 0.8  
- random_state: 42  
- n_jobs: -1  

**Ridge**
- alpha: 10.0  
- random_state: 42  

### 5) METRIKEN

**Baseline (Naive, Lag_15min auf Test-Set)**
- Test RMSE: 576.46  
- Test MAE: 441.33  
- Test R²: 0.9927  

**RandomForest**
- Train RMSE: 288.64  
- Test RMSE: 372.98  
- Train MAE: 211.03  
- Test MAE: 266.73  
- Train R²: 0.9984  
- Test R²: 0.9970  
- CV RMSE (Train-Set, 5 Folds): 399.72 ± 54.88  
- CV R² (Mean): 0.9970  

**GradientBoosting**
- Train RMSE: 346.83  
- Test RMSE: 375.27  
- Train MAE: 258.77  
- Test MAE: 270.41  
- Train R²: 0.9977  
- Test R²: 0.9969  
- CV RMSE (Train-Set, 5 Folds): 371.24 ± 19.05  
- CV R² (Mean): 0.9974  

**XGBoost**
- Train RMSE: 361.34  
- Test RMSE: 386.65  
- Train MAE: 270.64  
- Test MAE: 280.83  
- Train R²: 0.9975  
- Test R²: 0.9967  
- CV RMSE (Train-Set, 5 Folds): 391.40 ± 26.58  
- CV R² (Mean): 0.9971  

**Ridge**
- Train RMSE: 377.23  
- Test RMSE: 392.32  
- Train MAE: 274.43  
- Test MAE: 277.09  
- Train R²: 0.9973  
- Test R²: 0.9966  
- CV RMSE (Train-Set, 5 Folds): 399.51 ± 39.34  
- CV R² (Mean): 0.9970  

### 6) ERGEBNISSE
- Bestes Modell: RandomForest (niedrigster Test-RMSE 372.98, bestes Test-R² 0.9970)  
- Verbesserung vs Baseline: Reduktion des Test-RMSE von 576.46 auf 372.98, ca. 35.3 % Verbesserung gegenüber der naiven Lag_15min-Baseline  
- Outputs:
  - `model_comparison.csv` mit allen Modellen und Metriken
  - Plots: `model_comparison_rmse.png`, `forecast_comparison.png`, `residuals.png`
  - `features_used.txt` mit vollständiger Feature-Liste

### 7) ÄNDERUNGEN ZU V3
- Was wurde geändert:
  - Strikte Feature-Selektion: Nur Stromverbrauch-Lags, Zeit- und einfache Kunden-/Binary-Features; alle Wetter-Features entfernt.
  - Modell-Set reduziert auf 4 Modelle (RandomForest, GradientBoosting, XGBoost, Ridge) bei gleicher CV-Logik.
  - Separates Output-Verzeichnis `V4` inkl. Feature-Liste der genutzten Variablen.  

- Warum:
  - Ziel: explizit testen, wie gut sich der Verbrauch nur aus eigener Historie und Kalenderinformationen prognostizieren lässt, ohne externe Wetterinformationen.[web:18]
  - Überprüfung, wie stark Wetter-Features wirklich zur Performance beitragen, indem eine „No-Weather“-Variante direkt mit V3 verglichen werden kann.[web:18]  

- Erwartung:
  - Etwas schlechtere Performance als V3 mit Wetter-Features, aber weiterhin deutlich besser als die naive Lag-Baseline.
  - Klarer Einblick, welchen Mehrwert Wetterdaten bringen und ob ein reduziertes, robusteres Feature-Set für bestimmte Anwendungen ausreichend ist.[web:18]

---
## V5: All Features ohne Lag_15min
**Datei:** `C:\...\modeling_v5_without_lag15.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00  
- Ende: 2024-12-31 23:45:00  
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen im Zeitraum) [web:18]

### 2) FEATURES & ZIEL
- Target: Stromverbrauch  
- Features: Alle numerischen Features aus `processed_merged_features.csv`, nach Entfernen von 2 Object-Spalten (`Datum (Lokal)`, `Zeit (Lokal)`) und `Lag_15min` (insgesamt 38 Features)  
- Feature-Kategorien:  
  - Stromverbrauch-Lags: 5 (ohne `Lag_15min`)  
  - Wetter-Features: 23  
  - Zeit-Features: 5 (`Monat`, `Wochentag`, `Quartal`, `Woche des Jahres`, `Tag des Jahres`)  
  - Sonstige: 5  

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112192 Samples, Test: 28048 Samples)  
- Methode: Zeitlich (chronologischer Split)  
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set, mit RMSE und R² pro Fold [web:18]

### 4) MODELLE

**RandomForest**  
- n_estimators: 100  
- max_depth: 15  
- min_samples_split: 20  
- min_samples_leaf: 10  
- max_features: "sqrt"  
- random_state: 42  
- n_jobs: -1  

**GradientBoosting**  
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_samples_split: 20  
- min_samples_leaf: 10  
- subsample: 0.8  
- random_state: 42  

**XGBoost**  
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_child_weight: 10  
- subsample: 0.8  
- colsample_bytree: 0.8  
- random_state: 42  
- n_jobs: -1  

**Ridge**  
- alpha: 10.0  
- random_state: 42  

### 5) METRIKEN

**Baseline (Lag_30min auf Test-Set)**  
- Test RMSE: 1049.07  
- Test MAE: 808.66  
- Test R²: 0.9759  

**RandomForest**  
- Train RMSE: 289.07  
- Test RMSE: 405.04  
- Train MAE: 212.94  
- Test MAE: 294.56  
- Train R²: 0.9984  
- Test R²: 0.9964  
- Overfitting (Train R² − Test R²): 0.0020  
- CV RMSE (Train-Set, 5 Folds): 502.47 ± 165.45  
- CV R² (Mean): 0.9949  

**GradientBoosting**  
- Train RMSE: 321.20  
- Test RMSE: 350.22  
- Train MAE: 236.86  
- Test MAE: 250.02  
- Train R²: 0.9981  
- Test R²: 0.9973  
- Overfitting (Train R² − Test R²): 0.0007  
- CV RMSE (Train-Set, 5 Folds): 348.69 ± 18.68  
- CV R² (Mean): 0.9977  

**XGBoost**  
- Train RMSE: 332.54  
- Test RMSE: 357.49  
- Train MAE: 247.33  
- Test MAE: 258.41  
- Train R²: 0.9979  
- Test R²: 0.9972  
- Overfitting (Train R² − Test R²): 0.0007  
- CV RMSE (Train-Set, 5 Folds): 361.16 ± 23.35  
- CV R² (Mean): 0.9975  

**Ridge**  
- Train RMSE: 369.55  
- Test RMSE: 973.20  
- Train MAE: 269.16  
- Test MAE: 324.42  
- Train R²: 0.9974  
- Test R²: 0.9793  
- Overfitting (Train R² − Test R²): 0.0182  
- CV RMSE (Train-Set, 5 Folds): 394.78 ± 18.98  
- CV R² (Mean): 0.9971  

### 6) ERGEBNISSE
- Bestes Modell: GradientBoosting (niedrigster Test-RMSE 350.22, guter Test-MAE 250.02, bestes Test-R² 0.9973)  
- Verbesserung vs Baseline: Test-RMSE von 1049.07 (Lag_30min) auf 350.22 (GradientBoosting), ca. 66.6 % Reduktion des RMSE auf dem Test-Set  
- Outputs:
  - `model_comparison.csv` mit allen Modellen, Metriken und Overfitting-Score
  - Plots: `model_comparison.png` (RMSE & R²), `overfitting_comparison.png`, `forecast_comparison.png`, `residuals.png`, `feature_importance.png` (falls Feature Importances verfügbar)
  - `features_used.txt` mit Auflistung der genutzten Features nach Kategorien
  - `feature_importance.csv` mit Importances des besten Modells  

### 7) ÄNDERUNGEN ZU V4
- Was wurde geändert:
  - Wechsel von einem reduzierten Feature-Set (nur Lags + Zeit, keine Wetterdaten) zu einem vollen Feature-Set mit allen verfügbaren Variablen, außer `Lag_15min`.  
  - Baseline-Definition angepasst: statt `Lag_15min` nun `Lag_30min` als naive Referenz.  
  - Erweiterte Auswertung um expliziten Overfitting-Score (Train R² − Test R²) und Feature-Importance-Analysen für das beste Modell.  

- Warum:
  - Ziel ist es zu testen, wie gut das Modell mit allen Informationen (inkl. Wetter) performt, wenn der extrem starke Prädiktor `Lag_15min` entfernt wird, um eine realistischere Aussage über den Mehrwert der übrigen Features zu erhalten. [web:18]  
  - Overfitting-Messung und Feature-Importance helfen, Modellkomplexität und wichtigsten Treiber besser zu verstehen. [web:18]  

- Erwartung:
  - Besser als die Lag_30min-Baseline, aber tendenziell etwas schwächer als Versionen, die `Lag_15min` nutzen, dafür interpretierbarer bezüglich Wetter- und Zusatzfeatures. [web:18]  
  - Sehr kleines Overfitting bei GradientBoosting/XGBoost zeigt gute Generalisierung und robuste Nutzung der zusätzlichen Informationen. [web:18]

## V6: Echtes Forecasting 30min voraus
**Datei:** `C:\...\modeling_v6_forecast_30min.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00  
- Ende: 2024-12-31 23:45:00  
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen im Zeitraum; nach Target‑Shift: 140238 Samples) [web:44]

### 2) FEATURES & ZIEL
- Target: Stromverbrauch 30 Minuten in der Zukunft (Shift um 2 Time Steps nach vorne, `FORECAST_HORIZON_STEPS = 2`, `FORECAST_HORIZON_NAME = "30min"`). [web:44]  
- Features: Alle numerischen Features aus `processed_merged_features.csv` (39 Features nach Entfernen von 2 Object-Spalten).  
- Feature-Kategorien:  
  - Stromverbrauch-Lags: 6  
  - Wetter-Features: 23  
  - Zeit-Features: 5 (`Monat`, `Wochentag`, `Quartal`, `Woche des Jahres`, `Tag des Jahres`)  
  - Sonstige: 5  

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112190 Samples, Test: 28048 Samples).  
- Methode: Zeitlich (chronologischer Split; Train von 2021-01-01 00:00 bis 2024-03-14 18:15, Test von 2024-03-14 18:30 bis 2024-12-31 23:15).  
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set, Metriken RMSE und R² pro Fold. [web:45]

### 4) MODELLE

**RandomForest**  
- n_estimators: 100  
- max_depth: 15  
- min_samples_split: 20  
- min_samples_leaf: 10  
- max_features: "sqrt"  
- random_state: 42  
- n_jobs: -1  

**GradientBoosting**  
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_samples_split: 20  
- min_samples_leaf: 10  
- subsample: 0.8  
- random_state: 42  

**XGBoost**  
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_child_weight: 10  
- subsample: 0.8  
- colsample_bytree: 0.8  
- random_state: 42  
- n_jobs: -1  

**Ridge**  
- alpha: 10.0  
- random_state: 42  

### 5) METRIKEN

**Baseline (Persistence Forecast, Lag_15min auf Test-Set)**  
- Test RMSE: 1504.82  
- Test MAE: 1149.46  
- Test R²: 0.9504  

**RandomForest**  
- Train RMSE: 426.02  
- Test RMSE: 584.51  
- Train MAE: 320.43  
- Test MAE: 427.63  
- Train R²: 0.9966  
- Test R²: 0.9925  
- Overfitting (Train R² − Test R²): 0.0041  
- CV RMSE (Train-Set, 5 Folds): 653.47 ± 125.15  
- CV R² (Mean): 0.9918  

**GradientBoosting**  
- Train RMSE: 554.56  
- Test RMSE: 605.95  
- Train MAE: 422.46  
- Test MAE: 446.54  
- Train R²: 0.9942  
- Test R²: 0.9920  
- Overfitting: 0.0023  
- CV RMSE (Train-Set, 5 Folds): 619.65 ± 42.30  
- CV R² (Mean): 0.9927  

**XGBoost**  
- Train RMSE: 562.26  
- Test RMSE: 613.90  
- Train MAE: 429.24  
- Test MAE: 453.73  
- Train R²: 0.9941  
- Test R²: 0.9918  
- Overfitting: 0.0023  
- CV RMSE (Train-Set, 5 Folds): 631.22 ± 50.15  
- CV R² (Mean): 0.9925  

**Ridge**  
- Train RMSE: 818.27  
- Test RMSE: 1916.36  
- Train MAE: 623.69  
- Test MAE: 712.70  
- Train R²: 0.9874  
- Test R²: 0.9196  
- Overfitting: 0.0678  
- CV RMSE (Train-Set, 5 Folds): 966.08 ± 153.22  
- CV R² (Mean): 0.9823  

### 6) ERGEBNISSE
- Bestes Modell: RandomForest (niedrigster Test-RMSE 584.51, bestes Test-R² 0.9925, moderates Overfitting).  
- Verbesserung vs Baseline: Test-RMSE von 1504.82 (Persistence mit Lag_15min) auf 584.51 (RandomForest), ca. 61.2 % Reduktion des RMSE bei 30‑Minuten‑Forecast. [web:39][web:44]  
- Outputs:
  - `model_comparison.csv` mit allen Modellen, Test-/CV-Metriken und Overfitting-Score.  
  - Plots: `model_comparison.png` (RMSE & R²), `overfitting_comparison.png`, `forecast_comparison.png` (7 Tage), `residuals.png`, `feature_importance.png` (falls Feature Importances verfügbar).  
  - `feature_importance.csv` mit Importances des besten Modells.  

### 7) ÄNDERUNGEN ZU V5
- Was wurde geändert:
  - ECHTES Forecasting: Das Target wird um 2 Zeitschritte nach vorne geshiftet, sodass das Modell den Verbrauch 30 Minuten in der Zukunft vorhersagt, statt den aktuellen/nahe aktuellen Verbrauch zu reprojizieren. [web:44]  
  - Feature-Set: Alle Features inklusive `Lag_15min` und Wetterdaten bleiben erhalten (kein Entfernen von dominanten Lags).  
  - Baseline: Persistence-Ansatz mit `Lag_15min` als Vergleich für 30‑Minuten‑Forecast.  

- Warum:
  - Ziel ist, ein realistisches kurzfristiges Lastprognose-Setting (Very Short-Term Load Forecast, 30 Minuten ahead) abzubilden, wie es in der Literatur als wichtige Kategorie für Netzbetrieb und Regelung beschrieben wird. [web:44][web:39]  
  - Durch die Kombination aus vollständigem Feature-Set, Zeitreihen-CV und Persistance-Baseline lässt sich objektiv bewerten, wie viel Mehrwert die ML-Modelle für 30‑Minuten‑Horizonte liefern. [web:39][web:45]  

- Erwartung:
  - Deutlich schlechtere Metriken als bei „Jetzigkeits“-Forecasts (V1–V5), aber starke relative Verbesserung gegenüber Persistence; Performance fällt mit wachsendem Horizont typischerweise ab. [web:39][web:44]  
  - RandomForest als robuster Gewinner bei 30‑Minuten‑Forecast mit geringem Overfitting und guter Ausnutzung der Lag- und Wetterinformationen. [web:39]

## V6: Echtes Forecasting 30min voraus
**Datei:** `C:\...\modeling_v6_forecast_30min.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00  
- Ende: 2024-12-31 23:45:00  
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen im Zeitraum; nach Target‑Shift: 140238 Samples) [web:44]

### 2) FEATURES & ZIEL
- Target: Stromverbrauch 30 Minuten in der Zukunft (Shift um 2 Time Steps nach vorne, `FORECAST_HORIZON_STEPS = 2`, `FORECAST_HORIZON_NAME = "30min"`). [web:44]  
- Features: Alle numerischen Features aus `processed_merged_features.csv` (39 Features nach Entfernen von 2 Object-Spalten).  
- Feature-Kategorien:  
  - Stromverbrauch-Lags: 6  
  - Wetter-Features: 23  
  - Zeit-Features: 5 (`Monat`, `Wochentag`, `Quartal`, `Woche des Jahres`, `Tag des Jahres`)  
  - Sonstige: 5  

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112190 Samples, Test: 28048 Samples).  
- Methode: Zeitlich (chronologischer Split; Train von 2021-01-01 00:00 bis 2024-03-14 18:15, Test von 2024-03-14 18:30 bis 2024-12-31 23:15).  
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set, Metriken RMSE und R² pro Fold. [web:45]

### 4) MODELLE

**RandomForest**  
- n_estimators: 100  
- max_depth: 15  
- min_samples_split: 20  
- min_samples_leaf: 10  
- max_features: "sqrt"  
- random_state: 42  
- n_jobs: -1  

**GradientBoosting**  
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_samples_split: 20  
- min_samples_leaf: 10  
- subsample: 0.8  
- random_state: 42  

**XGBoost**  
- n_estimators: 100  
- learning_rate: 0.05  
- max_depth: 5  
- min_child_weight: 10  
- subsample: 0.8  
- colsample_bytree: 0.8  
- random_state: 42  
- n_jobs: -1  

**Ridge**  
- alpha: 10.0  
- random_state: 42  

### 5) METRIKEN

**Baseline (Persistence Forecast, Lag_15min auf Test-Set)**  
- Test RMSE: 1504.82  
- Test MAE: 1149.46  
- Test R²: 0.9504  

**RandomForest**  
- Train RMSE: 426.02  
- Test RMSE: 584.51  
- Train MAE: 320.43  
- Test MAE: 427.63  
- Train R²: 0.9966  
- Test R²: 0.9925  
- Overfitting (Train R² − Test R²): 0.0041  
- CV RMSE (Train-Set, 5 Folds): 653.47 ± 125.15  
- CV R² (Mean): 0.9918  

**GradientBoosting**  
- Train RMSE: 554.56  
- Test RMSE: 605.95  
- Train MAE: 422.46  
- Test MAE: 446.54  
- Train R²: 0.9942  
- Test R²: 0.9920  
- Overfitting: 0.0023  
- CV RMSE (Train-Set, 5 Folds): 619.65 ± 42.30  
- CV R² (Mean): 0.9927  

**XGBoost**  
- Train RMSE: 562.26  
- Test RMSE: 613.90  
- Train MAE: 429.24  
- Test MAE: 453.73  
- Train R²: 0.9941  
- Test R²: 0.9918  
- Overfitting: 0.0023  
- CV RMSE (Train-Set, 5 Folds): 631.22 ± 50.15  
- CV R² (Mean): 0.9925  

**Ridge**  
- Train RMSE: 818.27  
- Test RMSE: 1916.36  
- Train MAE: 623.69  
- Test MAE: 712.70  
- Train R²: 0.9874  
- Test R²: 0.9196  
- Overfitting: 0.0678  
- CV RMSE (Train-Set, 5 Folds): 966.08 ± 153.22  
- CV R² (Mean): 0.9823  

### 6) ERGEBNISSE
- Bestes Modell: RandomForest (niedrigster Test-RMSE 584.51, bestes Test-R² 0.9925, moderates Overfitting).  
- Verbesserung vs Baseline: Test-RMSE von 1504.82 (Persistence mit Lag_15min) auf 584.51 (RandomForest), ca. 61.2 % Reduktion des RMSE bei 30‑Minuten‑Forecast. [web:39][web:44]  
- Outputs:
  - `model_comparison.csv` mit allen Modellen, Test-/CV-Metriken und Overfitting-Score.  
  - Plots: `model_comparison.png` (RMSE & R²), `overfitting_comparison.png`, `forecast_comparison.png` (7 Tage), `residuals.png`, `feature_importance.png` (falls Feature Importances verfügbar).  
  - `feature_importance.csv` mit Importances des besten Modells.  

### 7) ÄNDERUNGEN ZU V5
- Was wurde geändert:
  - ECHTES Forecasting: Das Target wird um 2 Zeitschritte nach vorne geshiftet, sodass das Modell den Verbrauch 30 Minuten in der Zukunft vorhersagt, statt den aktuellen/nahe aktuellen Verbrauch zu reprojizieren. [web:44]  
  - Feature-Set: Alle Features inklusive `Lag_15min` und Wetterdaten bleiben erhalten (kein Entfernen von dominanten Lags).  
  - Baseline: Persistence-Ansatz mit `Lag_15min` als Vergleich für 30‑Minuten‑Forecast.  

- Warum:
  - Ziel ist, ein realistisches kurzfristiges Lastprognose-Setting (Very Short-Term Load Forecast, 30 Minuten ahead) abzubilden, wie es in der Literatur als wichtige Kategorie für Netzbetrieb und Regelung beschrieben wird. [web:44][web:39]  
  - Durch die Kombination aus vollständigem Feature-Set, Zeitreihen-CV und Persistance-Baseline lässt sich objektiv bewerten, wie viel Mehrwert die ML-Modelle für 30‑Minuten‑Horizonte liefern. [web:39][web:45]  

- Erwartung:
  - Deutlich schlechtere Metriken als bei „Jetzigkeits“-Forecasts (V1–V5), aber starke relative Verbesserung gegenüber Persistence; Performance fällt mit wachsendem Horizont typischerweise ab. [web:39][web:44]  
  - RandomForest als robuster Gewinner bei 30‑Minuten‑Forecast mit geringem Overfitting und guter Ausnutzung der Lag- und Wetterinformationen. [web:39]


## VERGLEICH ALLER VERSIONEN

| Version | Best Model | Test RMSE | Test R² | Besonderheit |
|---------|-----------|-----------|---------|--------------|
| V1      |           |           |         | Baseline     |
| V2      |           |           |         |              |
| V3      |           |           |         |              |

---

