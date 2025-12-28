# Energy-Forecasting-Anomaly-Detection-Basel
Wir entwickeln für Basel-Stadt ein datengetriebenes System zur Energie-Lastprognose und Anomalie-Erkennung. Auf Basis historischer Last-, Wetter- und Kalenderdaten sollen kurz- bis mittelfristige Forecasts gelingen und Abweichungen früh präventiv erkannt werden um Beschaffung, Netzbetrieb und Peak-Management verlässlicher und kosteneffizienter zu machen.


## Methodik
### CRISP-DM
Im Rahmen unseres ML-Projektes haben wir uns für die CRISP-DM-Zyklus-Vorgehensweise (Cross Industry Standard Process for Data Mining) entschieden. Dieses Vorgehensmodell gliedert den gesamten Datenanalyseprozess, von der Problemdefinition bis zur praktischen Nutzung des Modells, in sechs klar definierte Phasen.
CRISP-DM bietet uns eine strukturierte und iterative Vorgehensweise, um systematisch Daten zu verstehen, Modelle zu entwickeln und deren Nutzen im realen Umfeld zu evaluieren.

<img width="547" height="501" alt="image" src="https://github.com/user-attachments/assets/bd337948-9179-4135-807f-55f50c3d952d" />


### Business Understanding Data Understanding (Problemformalisierung)
- In dieser Phase wird das Problem fachlich verstanden und in eine formalisierte Aufgabenstellung für das Machine Learning übersetzt.
Zudem erfolgt eine erste Analyse der vorhandenen Datenquellen und -qualität.
### Data Preperation
- Die Daten werden aufbereitet, bereinigt und für das Modelltraining vorbereitet (Handling fehlender Werte,, Diskretisierung, Skalierung)
### Modeling
- Auswahl und Training des Modells (versch. Kriterien, Stärken & Schwächen und Parameter kennen)
### Evaluation
- Überprüfung der Modellleistung anhand geeigneter Metriken.
- Ziel ist es, die Ergebnisse kritisch zu bewerten, Probleme zu erkennen und den wirtschaftlichen Nutzen zu beurteilen.
### Deployment
- das entwickelte Modell oder die gewonnenen Erkenntnisse in die Praxis überführen.

(Das trainierte Prognosemodell wird als wiederverwendbare Python-Datei gespeichert. Ein separates Skript ruft das Modell regelmässig auf und erstellt täglich eine Vorhersage des Stromverbrauchs für den Folgetag. Die Prognosen werden automatisch in einer CSV-Datei oder als Diagramm gespeichert.
Dadurch ist eine einfache, reproduzierbare Nutzung der Ergebnisse möglich, ohne manuelle Eingriffe.

---

## Verwendung

### 1. Repository klonen
```bash
git clone https://github.com/Y-Akinci/Energy-Forecasting-Anomaly-Detection-Basel.git
cd Energy-Forecasting-Anomaly-Detection-Basel
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 3. Datenaufbereitung
Zuerst müssen die Rohdaten aufbereitet und zusammengeführt werden:
```bash
python Data_Preparation/data_preparation.py
```
Dies erstellt die Datei `data/processed_merged_features.csv` mit allen Features.

### 4. Datenexploration (optional)
Um die Daten zu verstehen und zu visualisieren:
```bash
python Business_und_Data_Understanding/Data_exploration/Data_exploration(06.12.2025).py
```
Die Visualisierungen werden im Ordner `Business_und_Data_Understanding/Data_exploration/Data/exploration_plots/` gespeichert.

### 5. Modelle trainieren
Es stehen verschiedene Forecasting-Ansätze zur Verfügung:

#### 5.1 Baseline-Modell (Prophet)
```bash
python Modeling/Forecasting/Yaren/baseline/prophet_model.py
```

#### 5.2 1-Step Forecast (15min voraus)
```bash
python Modeling/Forecasting/Yaren/1-Step_Forecast/1-step_forecast.py
```
Trainiert LGBM, XGBoost und Random Forest für 15-Minuten-Vorhersagen.

#### 5.3 Multi-Output Forecast (24h voraus, direkt)
```bash
python Modeling/Forecasting/Yaren/multi_output_forecast/modeling_multi_output.py
```

#### 5.4 Recursive Multi-Step Forecast (24h voraus, rekursiv)
```bash
python Modeling/Forecasting/Yaren/multi_step_forecast_recursive/multistep_forecast_recursive.py
```
Dies ist der empfohlene Ansatz für 24h-Prognosen.

### 6. Feature Importance analysieren
Nach dem Training können die wichtigsten Features analysiert werden:
```bash
python Modeling/Forecasting/Yaren/1-Step_Forecast/feature_importance_1-Step.py
python Modeling/Forecasting/Yaren/multi_output_forecast/feature_importance_multi_output.py
python Modeling/Forecasting/Yaren/multi_step_forecast_recursive/feature_importance_reursive_24h.py
```

### Projektstruktur
```
Energy-Forecasting-Anomaly-Detection-Basel/
├── data/                                    # Rohdaten und aufbereitete Daten
├── Data_Preparation/                       # Datenaufbereitung
│   └── data_preparation.py
├── Business_und_Data_Understanding/        # Datenexploration
│   └── Data_exploration/
├── Modeling/
│   └── Forecasting/Yaren/
│       ├── baseline/                       # Prophet Baseline
│       ├── 1-Step_Forecast/               # 15min Vorhersage
│       ├── multi_output_forecast/         # 24h direkt
│       └── multi_step_forecast_recursive/ # 24h rekursiv (empfohlen)
├── models/                                 # Trainierte Modelle
├── utils/                                  # Hilfsfunktionen
└── requirements.txt                        # Python Dependencies
```

### Hinweise
- Die Rohdaten müssen im `data/raw data/` Ordner liegen
- Trainierte Modelle werden automatisch im `models/` Ordner gespeichert
- Alle Zeitstempel werden in UTC verarbeitet und bei Bedarf in lokale Zeit (Europe/Zurich) konvertiert
- Das Projekt verwendet 15-Minuten-Intervalle für die Vorhersagen

