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

v8
## V8: 24h Forecast mit Top‑Features
**Datei:** `C:\...\modeling_v8_24h_forecast.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00  
- Ende: 2024-12-31 23:45:00  
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen; nach 24h‑Shift: 140144 Samples, Verlust 96 Zeilen durch Shift) [web:36]

### 2) FEATURES & ZIEL
- Target: Stromverbrauch 24 Stunden in der Zukunft (`FORECAST_HORIZON_STEPS = 96`, `FORECAST_HORIZON_NAME = "24h"`). [web:36]  
- Features: Top‑20‑Liste wie in V7/V6, davon 15 im Datensatz vorhanden (5 fehlen und werden ignoriert).  
- Verwendete Features (15):
  - Lags: `Lag_15min`, `Lag_30min`, `Lag_1h`, `Lag_24h`  
  - Wetter/Strahlung: `Diffusstrahlung; Zehnminutenmittel_lag15`, `Globalstrahlung; Zehnminutenmittel_lag15`, `Sonnenscheindauer; Zehnminutensumme_lag15`, `Lufttemperatur 2 m ü. Boden_lag15`, `relative Luftfeuchtigkeit_lag15`, `Lufttemperatur 2 m ü. Gras_lag15`, `Chilltemperatur_lag15`, `Böenspitze (Sekundenböe)_lag15`, `Lufttemperatur Bodenoberfläche_lag15`  
  - Kalender/Binary: `Stunde (Lokal)`, `IstArbeitstag`  

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112115 Samples, Test: 28029 Samples).  
- Methode: Zeitlich (chronologischer Split, gleiches Grundintervall wie V6/V7, aber mit 24h‑shifted Target).  
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set, Metriken RMSE und R² je Fold. [web:45]

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

*(XGBoost- und Ridge-Ergebnisse müsstest du analog zu RF/GB in die Doku nachtragen, sobald der Run komplett ist.)* [web:59]

### 5) METRIKEN

**Baseline (Lag_24h – Persistence)**  
- Test RMSE: 5449.78  
- Test MAE: 3777.78  
- Test R²: 0.3503  

**RandomForest**  
- Train RMSE: 2015.76  
- Test RMSE: 2589.03  
- Train MAE: 1365.89  
- Test MAE: 1761.84  
- Train R²: 0.9237  
- Test R²: 0.8534  
- Overfitting (Train R² − Test R²): 0.0704  
- CV RMSE (Train-Set, 5 Folds): 2687.81 ± 234.81  
- CV R² (Mean): 0.8622  

**GradientBoosting**  
- Train RMSE: 2488.04  
- Test RMSE: 2697.65  
- Train MAE: 1708.95  
- Test MAE: 1837.22  
- Train R²: 0.8838  
- Test R²: 0.8408  
- Overfitting: 0.0430  

*(CV‑Werte und XGBoost/Ridge analog einfügen, wenn vollständig.)* [web:36][web:59]

### 6) ERGEBNISSE
- Bestes Modell (bisheriger Stand): RandomForest mit niedrigstem Test-RMSE 2589.03 und höchstem Test-R² 0.8534 für 24h‑Forecast.  
- Verbesserung vs Baseline: Test-RMSE von 5449.78 (Lag_24h-Persistence) auf 2589.03 (RandomForest), etwa 52–53 % Reduktion des Fehlers trotz deutlich längerer Vorhersagehorizonte. [web:59][web:65]  
- Outputs:
  - `model_comparison.csv` mit allen Modellen und Metriken (inkl. Overfitting, CV).  
  - Plots: `model_comparison.png`, `overfitting_comparison.png`, `forecast_comparison.png` (14 Tage), `residuals.png`, `feature_importance.png`.  

### 7) ÄNDERUNGEN ZU V7
- Was wurde geändert:
  - Forecast-Horizont von 30 Minuten (V7) auf 24 Stunden erweitert (`shift` von −2 auf −96).  
  - Baseline angepasst: statt Lag_15min nun Lag_24h als Persistence-Benchmark.  
  - Visualisierung des Forecasts über 14 Tage, um Tagesmuster und Fehler über einen längeren Zeitraum zu sehen.  

- Warum:
  - Ziel ist die Abbildung eines „Short-Term / Day-Ahead Load Forecasting“-Szenarios, das in der Literatur als zentrale Aufgabe in der Energiewirtschaft beschrieben wird. [web:36][web:75]  
  - Vergleich zwischen sehr kurzfristen Horizonten (30 min) und Tagesprognosen zeigt, wie stark die Modellgüte mit dem Horizont abnimmt und wie viel Mehrwert ML gegenüber einfachen 24h‑Persistence-Benchmarks bringt. [web:36][web:65]  

- Erwartung:
  - Deutlich schlechtere absolute Metriken als bei 30‑Minuten‑Forecasts, aber signifikante relative Verbesserung gegenüber der 24h‑Baseline; R² im Bereich 0.8–0.9 ist für Day-Ahead-Lastprognose mit klassischen ML-Modellen typisch. [web:36][web:59]  
  - RandomForest als robustes Modell mit etwas stärkerem Overfitting, GradientBoosting als stabiler zweiter Kandidat mit geringerer Varianz über CV‑Splits. [web:59][web:67]

v9
## V9: 24h Forecast mit korrigierten Feature-Namen
**Datei:** `C:\...\modeling_v9_24h_corrected.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00  
- Ende: 2024-12-31 23:45:00  
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen; nach 24h‑Shift 140144 Samples, Verlust 96 Zeilen). [web:36]

### 2) FEATURES & ZIEL
- Target: Stromverbrauch 24 Stunden in der Zukunft (`FORECAST_HORIZON_STEPS = 96`, `FORECAST_HORIZON_NAME = "24h"`). [web:36]  
- Features: Top‑20‑Liste mit korrigierten Namen aus der CSV; 18 davon im Datensatz vorhanden (2 fehlen).  
- Verwendete Features (18):  
  - Lags: `Lag_15min`, `Lag_30min`, `Lag_1h`, `Lag_24h`  
  - Kunden: `Freie Kunden_Lag_15min`, `Grundversorgte Kunden_Lag_15min`  
  - Wetter/Strahlung: `Diffusstrahlung; Zehnminutenmittel_lag15`, `Globalstrahlung; Zehnminutenmittel_lag15`, `Sonnenscheindauer; Zehnminutensumme_lag15`, `Lufttemperatur 2 m ü. Boden_lag15`, `relative Luftfeuchtigkeit_lag15`, `Lufttemperatur 2 m ü. Gras_lag15`, `Chilltemperatur_lag15`, `Böenspitze (Sekundenböe)_lag15`, `Lufttemperatur Bodenoberfläche_lag15`  
  - Kalender/Binary: `Stunde (Lokal)`, `IstArbeitstag`, `IstSonntag`  

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112115 Samples, Test: 28029 Samples).  
- Methode: Zeitlich (chronologischer Split mit 24h‑shifted Target).  
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set, RMSE und R² pro Fold. [web:45]

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

**Baseline (Lag_24h – Persistence)**  
- Test RMSE: 5449.78  
- Test MAE: 3777.78  
- Test R²: 0.3503  

**RandomForest**  
- Train RMSE: 1721.79  
- Test RMSE: 2200.19  
- Train MAE: 1137.57  
- Test MAE: 1532.64  
- Train R²: 0.9444  
- Test R²: 0.8941  
- Overfitting (Train R² − Test R²): 0.0503  
- CV RMSE (Train-Set, 5 Folds): 2307.86 ± 152.75  
- CV R² (Mean): 0.8987  

**XGBoost**  
- Train RMSE: 2142.89  
- Test RMSE: 2223.34  
- Train MAE: 1447.99  
- Test MAE: 1557.60  
- Train R²: 0.9138  
- Test R²: 0.8919  
- Overfitting: 0.0219  
- CV RMSE (Train-Set, 5 Folds): 2274.26 ± 138.91  
- CV R² (Mean): 0.9016  

**GradientBoosting**  
- Train RMSE: 2136.25  
- Test RMSE: 2228.09  
- Train MAE: 1450.98  
- Test MAE: 1561.29  
- Train R²: 0.9143  
- Test R²: 0.8914  
- Overfitting: 0.0229  
- CV RMSE (Train-Set, 5 Folds): 2275.14 ± 140.74  
- CV R² (Mean): 0.9015  

**Ridge**  
- Train RMSE: 2986.90  
- Test RMSE: 2974.57  
- Train MAE: 2123.90  
- Test MAE: 2087.95  
- Train R²: 0.8325  
- Test R²: 0.8064  
- Overfitting: 0.0261  
- CV RMSE (Train-Set, 5 Folds): 3219.96 ± 336.97  
- CV R² (Mean): 0.8029  

### 6) ERGEBNISSE
- Bestes Modell: RandomForest (niedrigster Test-RMSE 2200.19, bestes Test-R² 0.8941; etwas stärkeres Overfitting als GB/XGBoost, aber beste Gesamtgüte).  
- Verbesserung vs Baseline: Test-RMSE von 5449.78 (Lag_24h-Persistence) auf 2200.19 (RandomForest), ca. 59.6 % Fehlerreduktion für 24‑Stunden‑Forecast. [web:36][web:59]  
- Outputs:
  - `model_comparison.csv`, Overfitting- und CV-Metriken je Modell.  
  - Plots: `model_comparison.png`, `overfitting_comparison.png`, `forecast_comparison.png` (14 Tage), `residuals.png`, `feature_importance.png`.  
  - `feature_importance.csv` mit Importances des besten Modells.  

### 7) ÄNDERUNGEN ZU V8
- Was wurde geändert:
  - Korrigierte Featurenamen entsprechend der tatsächlichen Spalten in der CSV (z. B. `Freie Kunden_Lag_15min`, `Grundversorgte Kunden_Lag_15min`, `IstSonntag`); dadurch mehr relevante Features (18 statt 15) im Modell.  
  - Gleicher 24h‑Forecast-Horizont und gleiche Baseline, aber verbesserte Feature-Mapping-Qualität.  

- Warum:
  - Ziel: Konsistentes Feature-Set über alle Versionen hinweg mit korrekter Benennung, um Interpretation und Wiederverwendbarkeit zu verbessern und keine wichtigen Kunden-/Kalendermerkmale zu verlieren. [web:54][web:61]  

- Erwartung:
  - Bessere Performance und stabilere CV-Werte als in V8 durch vollständigere Top‑Feature-Selektion; R² um 0.89 ist für Day-Ahead-Lastprognose mit klassischen ML-Modellen realistisch. [web:36][web:59]

v10
## V10: 1h Forecast ohne dominante Features
**Datei:** `C:\...\modeling_v10_1h_no_dominant_features.py`

### 1) ZEITFENSTER
- Start: 2021-01-01 00:00:00  
- Ende: 2024-12-31 23:45:00  
- Dauer: 4 Jahre (15‑Minuten-Daten, 140240 Zeilen; nach 1h‑Shift 140236 Samples). [web:36]

### 2) FEATURES & ZIEL
- Target: Stromverbrauch 1 Stunde in der Zukunft (`FORECAST_HORIZON_STEPS = 4`, `FORECAST_HORIZON_NAME = "1h"`). [web:36]  
- Entfernte dominante Features: `Lag_15min`, `Lag_30min`, Freie Kunden, Grundversorgte Kunden.  
- Verwendete Features (18):  
  - Lags: `Lag_1h`, `Lag_24h`, `Diff_15min`  
  - Wetter/Strahlung: `Diffusstrahlung; Zehnminutenmittel_lag15`, `Globalstrahlung; Zehnminutenmittel_lag15`, `Sonnenscheindauer; Zehnminutensumme_lag15`, `Lufttemperatur 2 m ü. Boden_lag15`, `relative Luftfeuchtigkeit_lag15`, `Lufttemperatur 2 m ü. Gras_lag15`, `Chilltemperatur_lag15`, `Böenspitze (Sekundenböe)_lag15`, `Lufttemperatur Bodenoberfläche_lag15`  
  - Kalender/Binary: `Stunde (Lokal)`, `IstArbeitstag`, `IstSonntag`, `Monat`, `Wochentag`, `Quartal`  

### 3) TRAIN/TEST SPLIT
- Ratio: 80/20 (Train: 112188 Samples, Test: 28048 Samples).  
- Methode: Zeitlich (chronologischer Split).  
- Cross-Validation: TimeSeriesSplit mit 5 Folds auf dem Train-Set (RMSE und R² je Fold). [web:45]

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

**Baseline (Lag_1h – Persistence)**  
- Test RMSE: 3681.75  
- Test MAE: 2794.25  
- Test R²: 0.7034  

**RandomForest**  
- Train RMSE: 653.62  
- Test RMSE: 924.83  
- Train MAE: 484.46  
- Test MAE: 653.26  
- Train R²: 0.9920  
- Test R²: 0.9813  
- Overfitting (Train R² − Test R²): 0.0107  
- CV RMSE (Train-Set, 5 Folds): 1040.06 ± 258.59  
- CV R² (Mean): 0.9787  

**GradientBoosting**  
- Train RMSE: 855.31  
- Test RMSE: 938.20  
- Train MAE: 636.46  
- Test MAE: 686.21  
- Train R²: 0.9863  
- Test R²: 0.9807  
- Overfitting: 0.0055  
- CV RMSE (Train-Set, 5 Folds): 930.65 ± 76.81  
- CV R² (Mean): 0.9835  

**XGBoost**  
- Train RMSE: 839.74  
- Test RMSE: 916.39  
- Train MAE: 630.52  
- Test MAE: 674.55  
- Train R²: 0.9868  
- Test R²: 0.9816  
- Overfitting: 0.0051  
- CV RMSE (Train-Set, 5 Folds): 911.60 ± 85.44  
- CV R² (Mean): 0.9841  

**Ridge**  
- Train RMSE: 1877.93  
- Test RMSE: 1898.68  
- Train MAE: 1379.58  
- Test MAE: 1392.87  
- Train R²: 0.9338  
- Test R²: 0.9211  
- Overfitting: 0.0127  
- CV RMSE (Train-Set, 5 Folds): 2094.94 ± 415.32  
- CV R² (Mean): 0.9148  

### 6) ERGEBNISSE
- Bestes Modell: XGBoost (niedrigster Test-RMSE 916.39, bestes Test-R² 0.9816, sehr geringes Overfitting).  
- Verbesserung vs Baseline: Test-RMSE von 3681.75 (Lag_1h-Persistence) auf 916.39 (XGBoost), ca. 75.1 % Fehlerreduktion bei 1‑Stunden‑Forecast ohne dominante Lags und Kundenfeatures. [web:36][web:77][web:94]  
- Outputs:
  - `model_comparison.csv` mit Test-/CV-Metriken und Overfitting je Modell.  
  - Plots: `model_comparison.png`, `overfitting_comparison.png`, `forecast_comparison.png` (7 Tage), `residuals.png`, `feature_importance.png`.  

### 7) ÄNDERUNGEN ZU V6/V7
- Was wurde geändert:
  - Forecast-Horizont auf 1 h festgelegt, aber bewusst stärkste kurzfristige Prädiktoren entfernt (`Lag_15min`, `Lag_30min`, Kunden-Features), um Modellleistung unter „schwierigeren“ Bedingungen zu testen. [web:86][web:88]  
  - Fokus auf Kombination aus mittel-/langfristigen Lags (`Lag_1h`, `Lag_24h`), Wetter- und Kalendermerkmalen, um strukturelle Muster statt triviale Autokorrelation auszunutzen. [web:88][web:92]  

- Warum:
  - Ziel: Einschätzen, wie gut die Modelle ohne dominante Autoregressions-Features performen und wie viel Mehrwert Wetter- und Kalenderdaten allein liefern. [web:88][web:92]  

- Erwartung:
  - Geringfügig schlechter als Setups mit allen Lags, aber immer noch deutlich besser als Persistence; hier bestätigen die Ergebnisse, dass auch ohne sehr nahes Lag starke 1h‑Forecasts möglich sind, insbesondere mit Gradient-Boosting-Modellen. [web:77][web:88]

v11
trainingszeit gändert 2023-2024
======================================================================
           Model   Test_RMSE    Test_MAE  Test_R2  Overfitting     CV_RMSE     CV_Std
         XGBoost  846.829183  628.491663 0.985248     0.000190 1030.238253 141.845401
GradientBoosting  866.542103  640.464470 0.984553     0.000273 1052.338172 149.237431
    RandomForest  907.070517  645.214355 0.983075     0.007835 1298.557508 449.030678
           Ridge 1858.653924 1368.936498 0.928936    -0.001527 1909.917662  29.272201

Bestes Modell: XGBoost
Test RMSE: 846.83
Test R²:   0.9852
Overfitting: 0.0002

v12
zeitperiode 2015 bis 2024
======================================================================
VERGLEICH - 1H FORECAST OHNE DOMINANTE FEATURES
======================================================================
           Model   Test_RMSE    Test_MAE  Test_R2  Overfitting     CV_RMSE     CV_Std
GradientBoosting 1147.349126  857.349276 0.973264     0.011367 1185.583612  80.847348
         XGBoost 1155.787167  869.177657 0.972870     0.011084 1175.373847  69.082291
    RandomForest 1196.086997  889.263392 0.970945     0.019139 1270.800642 130.476269
           Ridge 2920.290038 2244.534614 0.826798     0.018823 3306.349356 312.446031

Bestes Modell: GradientBoosting
Test RMSE: 1147.35
Test R²:   0.9733
Overfitting: 0.0114

Baseline RMSE:     3814.82
Best Model RMSE:   1147.35
Verbesserung:      69.9%

v13
train zu 70/30

======================================================================
VERGLEICH - 1H FORECAST OHNE DOMINANTE FEATURES
======================================================================
           Model   Test_RMSE    Test_MAE  Test_R2  Overfitting     CV_RMSE     CV_Std
         XGBoost 1183.520664  897.499843 0.972485     0.011877 1186.523755  76.803095
GradientBoosting 1195.197414  893.377092 0.971939     0.012938 1179.273372  81.794787
    RandomForest 1222.539861  921.378238 0.970641     0.019649 1281.544070 118.503987
           Ridge 2937.120180 2249.743186 0.830542     0.013244 3549.981794 581.452826

Bestes Modell: XGBoost
Test RMSE: 1183.52
Test R²:   0.9725
Overfitting: 0.0119

Baseline RMSE:     3867.15
Best Model RMSE:   1183.52
Verbesserung:      69.4%

v14
grid search mit rf
Beste Parameter:
  learning_rate: 0.08
  max_depth: 7
  min_samples_leaf: 10
  min_samples_split: 20
  n_estimators: 400
  random_state: 42
  subsample: 0.9
Bester CV Score (RMSE): 1014.18

GradientBoosting_GridSearch - Evaluation:
  Train RMSE: 710.11, MAE: 515.24, R²: 0.9934
  Test  RMSE: 925.45, MAE: 674.97, R²: 0.9832
  Overfitting: 0.0102

v15
# AKTUELL:
n_estimators=100      → 200-300 (mehr Bäume)
learning_rate=0.05    → 0.03-0.1 (testen)
max_depth=5           → 6-8 (tiefer)
XGBoost:
python

# AKTUELL:
n_estimators=100      → 200-300
learning_rate=0.05    → 0.03-0.1
max_depth=5           → 6-8
RandomForest:
python

# AKTUELL:
n_estimators=100      → 200 (mehr Bäume)
max_depth=15          → 20-25 (tiefer)
## VERGLEICH ALLER VERSIONEN

======================================================================
VERGLEICH - 1H FORECAST OHNE DOMINANTE FEATURES
======================================================================
           Model   Test_RMSE    Test_MAE  Test_R2  Overfitting     CV_RMSE     CV_Std
GradientBoosting 1047.458827  763.500356 0.978448     0.010409 1063.981391  71.860319
    RandomForest 1155.650162  859.375516 0.973766     0.019217 1241.093980 112.683866
         XGBoost 1209.862901  902.070071 0.971247     0.011843 1227.922827  66.981503
           Ridge 2937.120180 2249.743186 0.830542     0.013244 3549.981794 581.452826

Bestes Modell: GradientBoosting
Test RMSE: 1047.46
Test R²:   0.9784
Overfitting: 0.0104

| Version | Best Model | Test RMSE | Test R² | Besonderheit |
|---------|-----------|-----------|---------|--------------|
| V1      |           |           |         | Baseline     |
| V2      |           |           |         |              |
| V3      |           |           |         |              |

---
v16
recreating v13 weil beste werte, 70/30, 2015-2024

======================================================================
VERGLEICH - 1H FORECAST OHNE DOMINANTE FEATURES
======================================================================
           Model   Test_RMSE    Test_MAE  Test_R2  Overfitting     CV_RMSE     CV_Std
         XGBoost 1183.520664  897.499843 0.972485     0.011877 1186.523755  76.803095
GradientBoosting 1195.197414  893.377092 0.971939     0.012938 1179.273372  81.794787
    RandomForest 1222.539861  921.378238 0.970641     0.019649 1281.544070 118.503987
           Ridge 2937.120180 2249.743186 0.830542     0.013244 3549.981794 581.452826

Bestes Modell: XGBoost
Test RMSE: 1183.52
Test R²:   0.9725
Overfitting: 0.0119

v17
v16 ohne lag 1 h
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
======================================================================
VERGLEICH - 1H FORECAST OHNE DOMINANTE FEATURES
======================================================================
           Model   Test_RMSE    Test_MAE  Test_R2  Overfitting     CV_RMSE     CV_Std
GradientBoosting 1762.262747 1234.144288 0.938996     0.021455 1944.481281 141.905102
         XGBoost 1875.495500 1328.103831 0.930905     0.027770 2003.117556 154.306083
    RandomForest 2143.871868 1533.206550 0.909715     0.061949 2162.861251 182.950337
           Ridge 3416.383195 2688.744958 0.770728     0.021248 4112.430418 647.008884

Bestes Modell: GradientBoosting
Test RMSE: 1762.26
Test R²:   0.9390
Overfitting: 0.0215