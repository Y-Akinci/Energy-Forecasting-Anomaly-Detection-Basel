# ⚡ Energy Forecasting Basel

> Datengetriebenes System zur Energie-Lastprognose für Basel-Stadt basierend auf historischen Verbrauchs-, Wetter- und Kalenderdaten.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Überblick

Dieses Projekt entwickelt für IWB (Industrielle Werke Basel) ein Machine-Learning-System zur präzisen Vorhersage des Stromverbrauchs im Kanton Basel-Stadt. Die Prognosen ermöglichen optimierte Energiebeschaffung, effizienteres Netzmanagement und kosteneffizientere Planung.

### 🎯 Projektziele

- **15-Minuten-Prognosen**: Kurzfristige Verbrauchsvorhersagen im 15-Minuten-Takt
- **24-Stunden-Forecast**: Tagesprognosen für optimale Planungssicherheit
- **Hohe Genauigkeit**: R² > 0.998 auf Testdaten
- **Produktionsreif**: Reproduzierbare Pipeline von Datenaufbereitung bis Deployment

---

## 🚀 Key Features

✨ **Multi-Horizon Forecasting**
- 1-Step (15min voraus): MAE ~XXX kWh, R² X.XXX
- Recursive 24h: Komplette Tagesprognose mit rollierendem Forecast

🧠 **Ensemble von Modellen**
- LightGBM (Hauptmodell)
- XGBoost
- Random Forest
- Prophet (Baseline)

📊 **Umfangreicher Datensatz**
- **Stromverbrauch**: 481.959 Messwerte (2012-2025, 15-Minuten-Intervalle)
- **Wetterdaten**: 788.977 Messungen (2010-2024, 10-Minuten-Intervalle)
- **Features**: 60+ engineered Features (Lags, Kalender, Wetter)

🔧 **Production-Ready Pipeline**
- Automatische Datenaufbereitung
- Feature Engineering (Lags, Sin/Cos-Encoding, Weather-Lags)
- Modell-Persistierung mit joblib
- Umfassende Evaluation

---

## 📊 Ergebnisse

### Beste Modellperformance (1-Step Forecast)

| Modell | MAE (kWh) | RMSE (kWh) | R² | MAPE (%) |
|--------|-----------|------------|-----|----------|
| **LightGBM** | **XXX** | **XXX** | **X.XXX** | **X.XX** |
| XGBoost | XXX | XXX | X.XXX | X.XX |
| Random Forest | XXX | XXX | X.XXX | X.XX |
| Prophet (Baseline) | XXXX | XXXX | X.XXX | X.XX |

> **Interpretation**: Das Modell macht im Durchschnitt einen Fehler von nur XXX kWh bei einem mittleren Verbrauch von ~38.000 kWh - eine Abweichung von unter 1%.

### 24h Recursive Forecast

- **MAE (Ø)**: ~XXX kWh pro 15min-Intervall
- **R² (Ø)**: X.XXX
- Robuste Performance über gesamte Tagesprognose

Detaillierte Ergebnisse und Visualisierungen: → [Results.md](docs/Results.md)

---

## 🏗️ Projektstruktur

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
├── 📁 Archiv/                                 # Archivierte Dokumentation
│
├── 📄 requirements.txt                        # Python Dependencies
└── 📄 README.md                               # Diese Datei
```

---

## 🛠️ Installation & Verwendung

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

Trainiert LightGBM, XGBoost, Random Forest für 15-Minuten-Prognosen.

#### Option B: Recursive 24h Forecast (⭐ empfohlen für Tagesprognosen)

```bash
python Modeling/Forecasting/Yaren/multi_step_forecast_recursive/multistep_forecast_recursive.py
```

Erstellt rekursive 24h-Prognosen durch iteratives 1-Step-Forecasting.

#### Option C: Multi-Output Forecast (experimentell)

```bash
python Modeling/Forecasting/Yaren/multi_output_forecast/modeling_multi_output.py
```

Trainiert ein Modell, das direkt alle 96 Zeitpunkte (24h) vorhersagt.

#### Option D: Prophet Baseline

```bash
python Modeling/Forecasting/Yaren/baseline/prophet_model.py
```

Facebook Prophet als Benchmark für Zeitreihen-Forecasting.

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

## 📈 Methodik: CRISP-DM

Das Projekt folgt dem **CRISP-DM-Prozess** (Cross Industry Standard Process for Data Mining):

```
1. Business Understanding  →  2. Data Understanding  →  3. Data Preparation
                    ↑                                         ↓
                    ←  6. Deployment  ←  5. Evaluation  ←  4. Modeling
```

### Phasen

1. **Business Understanding**
   - Zielsetzung: Präzise Stromverbrauchsprognosen für IWB Basel
   - Nutzen: Kosteneffizienz, Planungssicherheit, optimierte Beschaffung

2. **Data Understanding**
   - 481.959 Stromverbrauchsmessungen (15min, 2012-2025)
   - 788.977 Wettermessungen (10min, 2010-2024)
   - Explorative Datenanalyse (EDA) mit 20+ Visualisierungen

3. **Data Preparation**
   - Zeitstempel-Synchronisation (UTC ↔ Europe/Zurich)
   - Interpolation: Wetter 10min → 15min
   - Feature Engineering: 60+ Features (Lags, Sin/Cos, Weather-Lags)
   - Datensplit: 70% Training, 30% Test (chronologisch)

4. **Modeling**
   - Ensemble-Approach: LightGBM, XGBoost, Random Forest
   - Hyperparameter-Tuning
   - Cross-Validation auf Zeitreihen

5. **Evaluation**
   - Metriken: MAE, RMSE, R², MAPE
   - Train/Test-Evaluation
   - 24h-Block-Evaluation für Robustheit

6. **Deployment** (geplant)
   - Automatisierte tägliche Prognosen
   - CSV/Plot-Export
   - Monitoring & Retraining

→ Detaillierte Beschreibung: [Data-Pipeline.md](docs/Data-Pipeline.md)

---

## 🎓 Lessons Learned & Besonderheiten

### 🔑 Kritische Erkenntnisse

1. **Zeitstempel sind komplex**: UTC vs. lokale Zeit, Sommerzeit-Problematik (52 fehlende Messwerte pro Jahr)
2. **15min ist Standard**: Stromhandel und IWB-Abrechnung basieren auf 15-Minuten-Intervallen
3. **Lag-Features sind essentiell**: `Lag_24h` ist der stärkste Prädiktor
4. **Wetter-Lags vermeiden Data Leakage**: Wetterfeatures werden 15min verzögert verwendet
5. **Sin/Cos-Encoding für zyklische Features**: Monat, Wochentag, Stunde werden trigonometrisch kodiert

### 💡 Spannende Code-Stellen

- **Rekursiver Forecast** (`multistep_forecast_recursive.py`): Wie das Modell iterativ 96 Schritte vorhersagt
- **Zeitstempel-Alignment** (`data_preparation.py`): UTC/Lokal-Konvertierung mit Sommerzeit-Handling
- **Feature Engineering** (`data_preparation.py`): Automatische Lag- und Weather-Feature-Erstellung

→ Technische Details: [Technical-Details.md](docs/Technical-Details.md)

---

## 📚 Dokumentation

| Datei | Inhalt |
|-------|--------|
| [**README.md**](README.md) | Projekt-Übersicht, Quick Start, Ergebnisse |
| [**Technical-Details.md**](docs/Technical-Details.md) | Technische Implementierung, Code-Erklärungen |
| [**Results.md**](docs/Results.md) | Detaillierte Modellergebnisse, Experimente |
| [**Data-Pipeline.md**](docs/Data-Pipeline.md) | Datenaufbereitung, Feature Engineering |

---

## 🤝 Beitragende

**Projektteam:**
- Yaren Akinci
- Haris Berbić

**Kontext:** Data Science Projekt, FHNW

---

## 📝 Hinweise

- **Datenquelle**: Die Rohdaten müssen im `data/raw data/` Ordner liegen
- **Modelle**: Trainierte Modelle werden in `models/` gespeichert (.joblib)
- **Zeitzone**: Alle Zeitstempel in UTC, Konvertierung nach Europe/Zurich für Features
- **Intervall**: 15 Minuten (Standard für Stromhandel)

---

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details.

---

## 🔗 Links

- [IWB Basel](https://www.iwb.ch/)
- [OpenData Basel-Stadt](https://opendata.swiss/de/dataset/kantonaler-stromverbrauch-netzlast)
- [MeteoSchweiz Daten](https://www.meteoschweiz.admin.ch/)

---

**⚡ Entwickelt mit Python, LightGBM und viel Kaffee ☕**
