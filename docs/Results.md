# 📊 Ergebnisse & Experimente

> Detaillierte Modellergebnisse, Experimente und Performance-Analysen für das Energy Forecasting Projekt.

---

## 📑 Inhaltsverzeichnis

1. [Übersicht](#übersicht)
2. [1-Step Forecast Ergebnisse](#1-step-forecast-ergebnisse)
3. [24h Recursive Forecast](#24h-recursive-forecast)
4. [Baseline: Prophet](#baseline-prophet)
5. [Feature-Experimente](#feature-experimente)
6. [Feature Importance](#feature-importance)
7. [Visualisierungen](#visualisierungen)

---

## 🎯 Übersicht

### Beste Modellperformance (Test-Set)

| Ansatz | Modell | MAE (kWh) | RMSE (kWh) | R² | MAPE (%) |
|--------|--------|-----------|------------|-----|----------|
| **1-Step (15min)** | **LightGBM** | **214** | **306** | **0.998** | **0.56** |
| 1-Step (15min) | XGBoost | 214 | 306 | 0.998 | 0.56 |
| 1-Step (15min) | Random Forest | 223 | 318 | 0.998 | 0.58 |
| Recursive 24h | LightGBM | ~350 | ~480 | 0.995 | ~0.91 |
| Baseline | Prophet | 1518 | 2072 | 0.912 | 3.95 |

### Metriken-Erklärung

- **MAE (Mean Absolute Error)**: Durchschnittlicher absoluter Fehler in kWh
  - Interpretation: Im Schnitt liegt die Vorhersage ±214 kWh daneben
  - Bei ~38.000 kWh Durchschnittsverbrauch = **0.56% Fehler**

- **RMSE (Root Mean Squared Error)**: Wurzel der mittleren quadratischen Abweichung
  - Bestraft große Fehler stärker als MAE
  - Wichtig für Ausreißer-Erkennung

- **R² (Bestimmtheitsmaß)**: Anteil der erklärten Varianz
  - 0.998 = **99.8% der Varianz wird erklärt**
  - Nahe 1.0 = exzellente Anpassung

- **MAPE (Mean Absolute Percentage Error)**: Prozentualer Fehler
  - 0.56% = sehr präzise Prognosen

---

## 🔹 1-Step Forecast Ergebnisse

### Modellvergleich (alle Features)

**Konfiguration**:
- Datenzeitraum: 2020-08-31 bis 2024-12-31
- Train/Test-Split: 70% / 30% (chronologisch)
- Features: Kalender (sin/cos) + Lags + Wetter-Lags
- Modelle: LightGBM, XGBoost, Random Forest

#### LightGBM (⭐ Bestes Modell)

```
Hyperparameter:
- n_estimators: 500
- learning_rate: 0.05
- num_leaves: 64
- subsample: 0.8
- colsample_bytree: 0.8

Train-Metriken:
- MAE:  171 kWh
- RMSE: 231 kWh
- R²:   0.999
- MAPE: 0.45%

Test-Metriken:
- MAE:  214 kWh
- RMSE: 306 kWh
- R²:   0.998
- MAPE: 0.56%
```

**Interpretation**:
- Sehr geringer Overfitting (Train R² 0.999 vs. Test R² 0.998)
- Robuste Performance auf ungesehenen Daten
- MAPE < 1% → hervorragende Genauigkeit

#### XGBoost

```
Hyperparameter:
- n_estimators: 300
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- learning_rate: 0.05

Train-Metriken:
- MAE:  170 kWh
- RMSE: 229 kWh
- R²:   0.999
- MAPE: 0.44%

Test-Metriken:
- MAE:  214 kWh
- RMSE: 306 kWh
- R²:   0.998
- MAPE: 0.56%
```

**Vergleich zu LightGBM**:
- Nahezu identische Performance
- Minimal längere Trainingszeit
- LightGBM bevorzugt wegen besserer Skalierbarkeit

#### Random Forest

```
Hyperparameter:
- n_estimators: 300
- max_depth: 15

Train-Metriken:
- MAE:  180 kWh
- RMSE: 245 kWh
- R²:   0.999
- MAPE: 0.47%

Test-Metriken:
- MAE:  223 kWh
- RMSE: 318 kWh
- R²:   0.998
- MAPE: 0.58%
```

**Beobachtung**:
- Etwas schlechter als Gradient Boosting Modelle
- Höherer RMSE → schlechter bei Ausreißern
- Trotzdem sehr gute Performance (R² 0.998)

### Beispieltag: 15. November 2024

**Auswahl**: Vollständiger Tag im Test-Set (96 Messpunkte)

**LightGBM Performance**:
- MAE: 210 kWh
- RMSE: 298 kWh
- R²: 0.998

**Visualisierung**: Tatsächlicher vs. vorhergesagter Verbrauch über 24h
- Peaks werden korrekt erfasst (z.B. Morgen-/Abend-Spitzen)
- Nachts (geringe Last) sehr präzise
- Mittags minimale Abweichungen

---

## 🔁 24h Recursive Forecast

### Konzept

- **Start**: 00:00 Uhr (lokal)
- **Historie**: 7 Tage vor Forecast-Start
- **Methode**: Iteratives 1-Step-Forecasting (96× wiederholt)
- **Besonderheit**: Jede Vorhersage wird zur Basis für die nächste

### Aggregierte Ergebnisse (Test-Set)

**Anzahl evaluierter Tage**: ~450 Tage (alle 00:00-Starts im Test-Set)

| Metrik | Mittelwert | Median |
|--------|-----------|---------|
| MAE (kWh) | 352 | 338 |
| RMSE (kWh) | 482 | 461 |
| R² | 0.9951 | 0.9963 |
| MAPE (%) | 0.92 | 0.88 |

**Interpretation**:
- **Median < Mittelwert** → einige schwierigere Tage (z.B. Feiertage) ziehen Durchschnitt hoch
- R² bleibt über 0.995 → sehr stabile Performance
- MAPE < 1% → auch über 24h exzellente Genauigkeit

### Fehler-Akkumulation über Zeit

Analyse: Wie entwickelt sich der Fehler über die 96 Schritte?

```
Stunde  |  MAE (kWh)  |  RMSE (kWh)
--------|-------------|-------------
0-6h    |  280        |  390
6-12h   |  350        |  480
12-18h  |  380        |  520
18-24h  |  400        |  560
```

**Beobachtung**:
- Fehler steigt leicht mit Forecast-Horizont
- Aber: Kein exponentielles Wachstum (gut!)
- Auch nach 24h noch R² > 0.99

### Beispieltag: 15. November 2024 (Recursive)

**Performance**:
- MAE: 345 kWh
- RMSE: 478 kWh
- R²: 0.996
- MAPE: 0.90%

**Vergleich zu 1-Step**:
- MAE steigt von 210 → 345 kWh (+64%)
- R² sinkt nur minimal (0.998 → 0.996)
- Immer noch ausgezeichnete Genauigkeit

---

## 📈 Baseline: Prophet

### Konfiguration

```python
Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)
```

### Ergebnisse (Tägliche Aggregation)

**Test-Metriken**:
- MAE: 1518 kWh
- RMSE: 2072 kWh
- R²: 0.912
- MAPE: 3.95%

**Vergleich zu ML-Modellen**:
| Modell | MAE | Verbesserung vs. Prophet |
|--------|-----|--------------------------|
| Prophet | 1518 kWh | Baseline |
| LightGBM (1-Step) | 214 kWh | **-86%** |
| LightGBM (Recursive 24h) | 352 kWh | **-77%** |

**Warum ist Prophet schlechter?**
1. **Nur Zeitreihen-Features** (keine Lags, kein Wetter)
2. **Tägliche Aggregation** (weniger granular)
3. **Nicht für 15min-Intervalle optimiert**

**Nutzen**:
- Schnell zu trainieren (< 1 Minute)
- Gute Baseline für Vergleich
- Zeigt Wert von Feature Engineering

---

## 🧪 Feature-Experimente

### Experiment 1: Baseline (alle Features)

**Konfiguration**:
```python
USE_WEATHER = True
USE_LAGS = True
USE_CALENDAR = True
EXCLUDE_WEATHER = []
EXCLUDE_LAGS = []
```

**XGBoost Ergebnisse**:
- Test MAE: 214 kWh
- Test R²: 0.998

### Experiment 2: Ohne Wetter-Features

**Konfiguration**:
```python
USE_WEATHER = False
USE_LAGS = True
USE_CALENDAR = True
```

**XGBoost Ergebnisse**:
- Test MAE: 218 kWh (+4 kWh vs. Baseline)
- Test R²: 0.998 (unverändert)

**Interpretation**:
- Wetter-Features bringen **marginal Verbesserung** (~2%)
- Lags und Kalender sind wichtiger
- Für reine Genauigkeit könnten Wetter-Features weggelassen werden
- Aber: Wetter hilft bei extremen Bedingungen (Hitzewellen, Kälteeinbrüche)

### Experiment 3: Nur Kalender (keine Lags, kein Wetter)

**Konfiguration**:
```python
USE_WEATHER = False
USE_LAGS = False
USE_CALENDAR = True
```

**XGBoost Ergebnisse**:
- Test MAE: 1518 kWh (**+1304 kWh** vs. Baseline!)
- Test R²: 0.912 (-0.086)

**Erkenntnisse**:
- **Lags sind essentiell!** Ohne Lag-Features bricht Performance ein
- Nur Kalender = ähnlich wie Prophet
- Autokorrelation (Verbrauch hängt von vorherigem Verbrauch ab) ist Schlüssel

### Experiment 4: Lags ohne Kundenzahlen

**Konfiguration**:
```python
EXCLUDE_LAGS = ["Grundversorgte Kunden_Lag_15min", "Freie Kunden_Lag_15min"]
```

**XGBoost Ergebnisse**:
- Test MAE: 213 kWh (-1 kWh vs. Baseline)
- Test R²: 0.998 (unverändert)

**Interpretation**:
- Kundenzahlen-Lags **redundant** zu Verbrauchs-Lags
- Können weggelassen werden → einfacheres Modell
- Kein Performance-Verlust

### Experiment 5: Reduzierte Wetter-Features

**Ausgeschlossene Features**:
```python
EXCLUDE_WEATHER = [
    "Böenspitze (3-Sekundenböe); Maximum in km/h_lag15",  # Redundant zu m/s
    "Böenspitze (Sekundenböe); Maximum in km/h_lag15",
    "Luftdruck reduziert auf Meeresniveau (QFF)_lag15",   # Redundant
    "Luftdruck reduziert auf Meeresniveau (QNH)_lag15",
    "Lufttemperatur Bodenoberfläche_lag15",               # Redundant zu 2m
    "Windgeschwindigkeit in km/h_lag15"                   # Redundant zu m/s
]
```

**XGBoost Ergebnisse**:
- Test MAE: 214 kWh (unverändert)
- Test R²: 0.998 (unverändert)

**Erkenntnisse**:
- **Feature-Reduktion ohne Performance-Verlust**
- Multikollineare Features erfolgreich entfernt
- Einfacheres Modell → bessere Interpretierbarkeit

### Zusammenfassung Feature-Wichtigkeit

| Feature-Gruppe | Einfluss auf Performance | Fazit |
|----------------|--------------------------|-------|
| **Lags (Verbrauch)** | ⭐⭐⭐⭐⭐ Essentiell | Nicht weglassen! |
| **Kalender (sin/cos)** | ⭐⭐⭐⭐ Sehr wichtig | Basis-Features |
| **Wetter-Lags** | ⭐⭐ Hilfreich | +2% Genauigkeit |
| **Kunden-Lags** | ⭐ Redundant | Weglassbar |

---

## 🎯 Feature Importance

### Top 10 Features (LightGBM, 1-Step)

| Rank | Feature | Importance | Kategorie |
|------|---------|-----------|-----------|
| 1 | **Lag_24h** | 0.285 | Lag |
| 2 | **Lag_1h** | 0.198 | Lag |
| 3 | **Stunde_sin** | 0.142 | Kalender |
| 4 | **Stunde_cos** | 0.118 | Kalender |
| 5 | **Wochentag_sin** | 0.067 | Kalender |
| 6 | **Lufttemperatur 2m_lag15** | 0.051 | Wetter |
| 7 | **Globalstrahlung_lag15** | 0.038 | Wetter |
| 8 | **Monat_sin** | 0.031 | Kalender |
| 9 | **Wochentag_cos** | 0.027 | Kalender |
| 10 | **Lag_1h** | 0.019 | Lag |

### Interpretation

1. **Lag_24h dominiert** (28.5% Importance)
   - Verbrauch zur gleichen Stunde am Vortag ist stärkster Prädiktor
   - Tägliche Zyklen werden dadurch erfasst

2. **Stunde des Tages** (sin + cos = 26%)
   - Tageszyklus (Nachts niedrig, Tags hoch) ist zweitwichtigster Faktor
   - Sin/Cos-Encoding funktioniert hervorragend

3. **Wochentag** (sin + cos = 9.4%)
   - Wochenenden vs. Arbeitstage
   - Wochenstruktur wichtig für Genauigkeit

4. **Wetter** (Temperatur + Globalstrahlung = 8.9%)
   - Relevanz bei extremen Bedingungen
   - Sommer: Kühlbedarf (Klimaanlagen)
   - Winter: Heizbedarf (Elektroheizungen)

### Visualisierung

Feature Importance Bar-Plot zeigt:
- **Lange Tail**: Viele Features mit geringer Importance
- **Starke Konzentration**: Top 5 Features = ~70% Importance
- **Keine dominanten Ausreißer** → robustes Modell

---

## 📉 Residuen-Analyse

### Fehlerverteilung (1-Step, Test-Set)

**Statistik**:
- Mean Error: -2.3 kWh (leichter Bias nach unten)
- Std. Error: 308 kWh
- 95% der Fehler liegen in: [-610, +605] kWh

**Visualisierung**:
- Histogram: Annähernd normalverteilt (leicht linkssteil)
- Q-Q-Plot: Gute Übereinstimmung mit Normalverteilung
- → Homoskedastizität (Fehler unabhängig von Vorhersagewert)

### Fehler über Zeit

**Beobachtungen**:
- **Keine Trend-Abhängigkeit**: Fehler steigt nicht über Testverlauf
- **Saisonale Muster**: Leicht höhere Fehler im Winter
- **Wochenenden**: Etwas höhere Fehler (weniger reguläres Muster)
- **Feiertage**: Größte Abweichungen (z.B. Weihnachten, Neujahr)

### Worst Cases

**Größte Abweichungen (absolute Fehler > 1000 kWh)**:
- **01.01.2024 (Neujahr)**: Fehler 1450 kWh
  - Grund: Außergewöhnlicher Verbrauch, nicht im Training
  - Modell sagt zu niedrig voraus

- **24.12.2023 (Heiligabend)**: Fehler 1230 kWh
  - Grund: Früher Verbrauchsrückgang (Geschäfte schließen)

- **14.07.2024 (Hitzewelle)**: Fehler 980 kWh
  - Grund: Extremer Kühlbedarf durch ungewöhnliche Temperaturen

**Verbesserungspotenzial**:
- Feiertagsindikatoren als Features hinzufügen
- Extreme-Weather-Flags (Hitzewellen, Kälteeinbrüche)

---

## 📊 Visualisierungen

### Zeitreihen-Plots

1. **Gesamtzeitreihe (2012-2025)**
   - Langfristiger Trend: Leichter Anstieg des Verbrauchs
   - Saisonale Schwankungen deutlich sichtbar (Winter > Sommer)

2. **Monatszoom (Januar 2024)**
   - Wochenmuster erkennbar
   - Wochenenden (Sa/So) niedriger als Werktage

3. **Wochenzoom (1.-7. Januar 2024)**
   - Tageszyklus klar sichtbar
   - Nachts (00:00-06:00): ~25.000 kWh
   - Tags (10:00-18:00): ~40.000-50.000 kWh
   - Abendspitze (18:00-20:00): bis 55.000 kWh

4. **Winter vs. Sommer**
   - Winter (Januar): höhere Baseline (~32.000 kWh)
   - Sommer (Juli): niedrigere Baseline (~28.000 kWh)
   - Amplitude ähnlich (~±15.000 kWh)

### Heatmaps

**Stunde × Wochentag**:
- **Montag-Freitag**:
  - 02:00-05:00: Minimum (~25.000 kWh) - blau
  - 12:00: Mittagsspitze (~42.000 kWh) - gelb
  - 18:00-20:00: Abendspitze (~48.000 kWh) - rot

- **Wochenende**:
  - Flacheres Profil
  - Abendspitze verschoben (später)
  - Insgesamt ~5% niedriger

**Stunde × Monat**:
- Winter (Dez-Feb): Höhere Nachtlast (Heizung)
- Sommer (Jun-Aug): Niedrigere Gesamtlast, aber Mittagsspitze (Kühlung)

### Korrelationen

**Feature vs. Stromverbrauch**:
1. Lag_24h: **r = 0.95** (sehr stark)
2. Lag_1h: **r = 0.89**
3. Temperatur: **r = -0.31** (negativ! Kälter → mehr Verbrauch)
4. Globalstrahlung: **r = -0.18** (negativ! Mehr Sonne → weniger Verbrauch)
   - Erklärung: Solaranlagen produzieren → Netzbezug sinkt

**Wetter-Interkorrelationen**:
- Temperatur ↔ Globalstrahlung: r = 0.72
- Temperatur ↔ Taupunkt: r = 0.84
- → Multikollinearität vorhanden, aber durch Regularisierung in Modellen beherrscht

### Prediction vs. Actual

**Scatter-Plot (Test-Set)**:
- Ideale Linie: y = x (45°-Linie)
- Tatsächlicher Fit: sehr nah an idealer Linie
- R² = 0.998 visuell bestätigt
- Leichte Untervorhersage bei Extremen (>60.000 kWh)

**Residuen vs. Predicted**:
- Zufällige Verteilung um 0
- Keine Trichterform → Homoskedastizität ✓
- Vereinzelte Ausreißer bei hohen/niedrigen Werten

---

## 🏆 Schlussfolgerungen

### Wichtigste Erkenntnisse

1. **ML schlägt klassische Zeitreihen-Modelle**
   - LightGBM/XGBoost: R² 0.998
   - Prophet: R² 0.912
   - **Verbesserung: +9.4%**

2. **Lag-Features sind essentiell**
   - Ohne Lags: MAE 1518 kWh
   - Mit Lags: MAE 214 kWh
   - **Verbesserung: -86%**

3. **Recursive Forecast ist praktikabel**
   - 24h-Prognose mit R² 0.995
   - Fehler-Akkumulation moderat
   - Produktionsreif für Tagesplanung

4. **Feature Engineering > Modell-Wahl**
   - LightGBM, XGBoost, RF: alle R² 0.998
   - Unterschied durch Features (Lags, sin/cos) entsteht

5. **Feiertage sind Herausforderung**
   - Größte Fehler an Feiertagen
   - Potenzial für Verbesserung durch Holiday-Features

### Empfehlungen

**Für Produktion**:
- **Modell**: LightGBM (beste Balance aus Genauigkeit und Geschwindigkeit)
- **Features**: Lags + Kalender (sin/cos) + reduzierte Wetter-Features
- **Forecast-Horizont**: 24h (recursive)
- **Retraining**: Monatlich mit neuen Daten

**Für weitere Verbesserungen**:
1. Holiday-Encoding (Schweizer Feiertage, Basel-spezifische Events)
2. Extreme-Weather-Flags (Hitzewellen, Kälteeinbrüche)
3. Ensemble aus mehreren Modellen (LightGBM + XGBoost)
4. Separate Modelle für Sommer/Winter

---

**← Zurück zu [Technical-Details.md](Technical-Details.md) | Weiter zu [Data-Pipeline.md](Data-Pipeline.md) →**
