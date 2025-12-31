# ⚡ Electricity Forecasting Basel

> Datengetriebenes System zur Energie-Lastprognose für Basel-Stadt basierend auf historischen Verbrauchs-, Wetter- und Kalenderdaten.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Überblick

Ein Machine-Learning-System zur präzisen Vorhersage des Stromverbrauchs im Kanton Basel-Stadt. Die Prognosen ermöglichen optimierte Energiebeschaffung, effizienteres Netzmanagement und kosteneffizientere Planung.

### Projektziele

- Kurzfristige und mittelfristige Verbrauchsvorhersagen im 15-Minuten-Takt
-	Aufbau eines konsistenten, reproduzierbaren Feature Sets aus den vorhandenen Daten
-	Ableitung zusätzlicher relevanter Merkmale und Ergänzung durch weitere relevante Datensätze
-	Training eines Regressionsmodells zur Vorhersage des Stromverbrauchs
-	Bewertung der Modelgüte anhand transparenter Metriken, erklärbaren Modellen und Grafiken

---

## Key Features

**Multi-Horizon Forecasting**
- 1-Step (15min voraus): MAE ~413 kWh, R² 0.994
- Recursive 24h: MAE ~858 kWh, R² 0.863

**Modell**
- LightGBM

**Umfangreicher Datensatz**
- **Stromverbrauch**: 481'959 Messwerte (2012-2025, 15-Minuten-Intervalle)
- **Wetterdaten**: 788'977 Messungen (2010-2024, 10-Minuten-Intervalle)
- **Features**: 40+ engineered Features (Lags, Kalender, Wetter)

**Production-Ready**
- Datenaufbereitung
- Feature Engineering (Lags, Sin/Cos-Encoding, Weather-Lags)
- Modell-Persistierung mit joblib
- Umfassende Evaluation

---

## Ergebnisse

### Modellperformance (LightGBM)

#### 1-Step Forecast (15min voraus)

| Datensatz | MAE (kWh) | RMSE (kWh) | R² | MAPE (%) |
|-----------|-----------|------------|-----|----------|
| **Train** | **298.15** | **394.22** | **0.9971** | **0.86%** |
| **Test** | **412.54** | **564.30** | **0.9935** | **1.19%** |

> **Interpretation**: Das Modell macht im Test-Datensatz einen durchschnittlichen Fehler von 413 kWh bei einem mittleren Verbrauch von ~35.000 kWh - eine Abweichung von nur 1,19%.

#### Multi-Output 24h Global

| Datensatz | MAE (kWh) | RMSE (kWh) | R² | MAPE (%) |
|-----------|-----------|------------|-----|----------|
| **Train** | **568.11** | **744.87** | **0.9897** | **1.63%** |
| **Test** | **1069.02** | **1551.08** | **0.9510** | **3.10%** |

Hier wird der gesamte 24-Stunden-Horizont gleichzeitig als Vektor vorhergesagt.

#### Multi-Output 24h Block (Start 00:00 lokal)

**Train (1108 Tage):**
- MAE (Mean): 701.25 kWh | RMSE (Mean): 866.82 kWh | R² (Mean): 0.9649 | MAPE (Mean): 2.03%
- MAE (Median): 649.74 kWh

**Test (475 Tage):**
- MAE (Mean): 1405.26 kWh | RMSE (Mean): 1723.25 kWh | R² (Mean): 0.7809 | MAPE (Mean): 4.07%
- MAE (Median): 1234.68 kWh

#### Recursive 24h Block (Start 00:00 lokal) ⭐ Empfohlen

**Train (1108 Tage):**
- MAE (Mean): 516.30 kWh | RMSE (Mean): 660.73 kWh | R² (Mean): 0.9575 | MAPE (Mean): 1.51%
- MAE (Median): 451.20 kWh

**Test (474 Tage):**
- MAE (Mean): 858.15 kWh | RMSE (Mean): 1073.77 kWh | R² (Mean): 0.8632 | MAPE (Mean): 2.48%
- MAE (Median): 677.15 kWh

> **💡 ML-Einschätzung**: Der rekursive Ansatz zeigt im Test-Datensatz eine deutlich bessere Generalisierung als der Multi-Output Block (MAPE 2.48% vs. 4.07%). Das Modell profitiert stark von der zeitlichen Abhängigkeit der Daten. Die R² Werte über 0.95 unterstreichen die hohe Güte des LightGBM-Modells für die Lastprognose in Basel.

Detaillierte Ergebnisse: → [Results.md](docs/Results.md)

---

## Projektstruktur

```
Energy-Forecasting-Anomaly-Detection-Basel/
│
├── 📁 data/                                    # Daten
│   ├── raw data/                              # Rohdaten (Strom + Wetter)
│   └── processed_merged_features.csv          # Aufbereiteter Datensatz
│
├── 📁 Data_Preparation/                       # Datenaufbereitung
│   └── data_preparation.py                    # Haupt-Pipeline
│
├── 📁 Business_und_Data_Understanding/        # EDA & Dokumentation
│   ├── Data_exploration/
│   │   └── Data_exploration(06.12.2025).py   # Explorative Datenanalyse
│   ├── business_understanding.md
│   └── data_understanding.md
│
├── 📁 Modeling/                               # Machine Learning
│   └── Forecasting/Yaren/
│       ├── baseline/                          # Prophet Baseline
│       ├── 1-Step_Forecast/                  # 15min Vorhersage
│       ├── multi_output_forecast/            # 24h Direct
│       └── multi_step_forecast_recursive/    # 24h Recursive (⭐ empfohlen)
│
├── 📁 models/                                 # Trainierte Modelle (.joblib)
├── 📁 utils/                                  # Helper-Funktionen
│                            
│
├── 📄 requirements.txt                        # Python Dependencies
└── 📄 README.md                               # Diese Datei
```

---

## Installation & Verwendung

### 1️⃣ Repository klonen

```bash
git clone https://github.com/Y-Akinci/Energy-Forecasting-Anomaly-Detection-Basel.git
cd Energy-Forecasting-Anomaly-Detection-Basel
```

### 2️⃣ Dependencies installieren

```bash
pip install -r requirements.txt
```

**Benötigte Pakete:**
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, lightgbm, prophet
- scipy, statsmodels, joblib

### 3️⃣ Daten aufbereiten

```bash
python Data_Preparation/data_preparation.py
```

**Was passiert hier?**
- Lädt Stromverbrauchs- und Wetterdaten
- Merged beide Datensätze auf 15-Minuten-Basis
- Erstellt Features (Lags, Kalender, Wetter-Lags)
- Speichert `data/processed_merged_features.csv`

⏱️ **Dauer**: ~2-3 Minuten

### 4️⃣ Datenexploration (optional)

```bash
python Business_und_Data_Understanding/Data_exploration/Data_exploration(06.12.2025).py
```

Erstellt 20+ Visualisierungen zur Datenanalyse:
- Zeitreihen-Plots (gesamt, monatlich, wöchentlich)
- Heatmaps (Stunde × Wochentag)
- Feature-Korrelationen
- Saisonale Dekomposition
- Autokorrelation (ACF)

### 5️⃣ Modelle trainieren

#### Option A: 1-Step Forecast (⚡ schnell, hohe Genauigkeit)

```bash
python Modeling/Forecasting/Yaren/1-Step_Forecast/1-step_forecast.py
```

Trainiert LightGBM für 15-Minuten-Prognosen.

#### Option B: Recursive 24h Forecast (⭐ empfohlen für Tagesprognosen)

```bash
python Modeling/Forecasting/Yaren/multi_step_forecast_recursive/multistep_forecast_recursive.py
```

Erstellt rekursive 24h-Prognosen durch iteratives 1-Step-Forecasting.

#### Option C: Multi-Output Forecast

```bash
python Modeling/Forecasting/Yaren/multi_output_forecast/modeling_multi_output.py
```

Trainiert ein Modell, das direkt alle 96 Zeitpunkte (24h) vorhersagt.

### 6️⃣ Feature Importance analysieren

```bash
python Modeling/Forecasting/Yaren/1-Step_Forecast/feature_importance_1-Step.py
```

Zeigt die wichtigsten Einflussgrößen auf den Stromverbrauch:
1. **Lag_24h** (Verbrauch vor 24h) - stärkster Prädiktor
2. **Lag_1h**, **Lag_15min** - kurzfristige Autokorrelation
3. **Stunde (sin/cos)** - Tageszyklus
4. **Wochentag (sin/cos)** - Wochenstruktur
5. **Temperatur**, **Globalstrahlung** - Wettereinfluss

---

## Methodik: CRISP-DM

Das Projekt folgt dem **CRISP-DM-Prozess** (Cross Industry Standard Process for Data Mining):

```
1. Business Understanding  →  2. Data Understanding  →  3. Data Preparation
                    ↑                                         ↓
                    ←  6. Deployment  ←  5. Evaluation  ←  4. Modeling
```

## Lessons Learned & Besonderheiten

### Kritische Erkenntnisse

1. **Zeitstempel sind komplex**: UTC vs. lokale Zeit, Sommerzeit-Problematik (52 fehlende Messwerte pro Jahr)
2. **15min ist Standard**: Stromhandel und IWB-Abrechnung basieren auf 15-Minuten-Intervallen
3. **Lag-Features sind essentiell**: `Lag_24h` ist der stärkste Prädiktor
4. **Wetter-Lags vermeiden Data Leakage**: Wetterfeatures werden 15min verzögert verwendet
5. **Sin/Cos-Encoding für zyklische Features**: Monat, Wochentag, Stunde werden trigonometrisch kodiert

### Spannende Code-Stellen

- **Rekursiver Forecast** (`multistep_forecast_recursive.py`): Wie das Modell iterativ 96 Schritte vorhersagt
- **Zeitstempel-Alignment** (`data_preparation.py`): UTC/Lokal-Konvertierung mit Sommerzeit-Handling
- **Feature Engineering** (`data_preparation.py`): Automatische Lag- und Weather-Feature-Erstellung

→ Technische Details: [Technical-Details.md](docs/Technical-Details.md)

---

## Dokumentation

| Datei | Inhalt |
|-------|--------|
| [**README.md**](README.md) | Projekt-Übersicht, Quick Start, Ergebnisse |
| [**Technical-Details.md**](docs/Technical-Details.md) | Technische Implementierung, Code-Erklärungen |
| [**Results.md**](docs/Results.md) | Detaillierte Modellergebnisse, Experimente |
| [**Data-Pipeline.md**](docs/Data-Pipeline.md) | Datenaufbereitung, Feature Engineering |

---

## Beitragende

**Projektteam:**
- Yaren Akinci
- Haris Salii
- kerem Akkaya

**Kontext:** ML Projekt, FHNW

---

## Hinweise

- **Datenquelle**: Die Rohdaten müssen im `data/raw data/` Ordner liegen
- **Modelle**: Trainierte Modelle werden in `models/` gespeichert (.joblib)
- **Zeitzone**: Alle Zeitstempel in UTC, Konvertierung nach Europe/Zurich für Features
- **Intervall**: 15 Minuten (Standard für Stromhandel)

---

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details.

---

## 🔗 Links

- [IWB Basel](https://www.iwb.ch/)
- [OpenData Basel-Stadt](https://opendata.swiss/de/dataset/kantonaler-stromverbrauch-netzlast)
- [MeteoSchweiz Daten](https://www.meteoschweiz.admin.ch/)

---

** Entwickelt mit Python und viel Kaffee **
