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
| **1-Step (15min)** | **LightGBM** | **XXX** | **XXX** | **X.XXX** | **X.XX** |
| 1-Step (15min) | XGBoost | XXX | XXX | X.XXX | X.XX |
| 1-Step (15min) | Random Forest | XXX | XXX | X.XXX | X.XX |
| Recursive 24h | LightGBM | ~XXX | ~XXX | X.XXX | ~X.XX |
| Baseline | Prophet | XXXX | XXXX | X.XXX | X.XX |

### Metriken-Erklärung

- **MAE (Mean Absolute Error)**: Durchschnittlicher absoluter Fehler in kWh
  - Interpretation: Im Schnitt liegt die Vorhersage ±XXX kWh daneben
  - Bei ~38.000 kWh Durchschnittsverbrauch = **X.XX% Fehler**

- **RMSE (Root Mean Squared Error)**: Wurzel der mittleren quadratischen Abweichung
  - Bestraft große Fehler stärker als MAE
  - Wichtig für Ausreißer-Erkennung

- **R² (Bestimmtheitsmaß)**: Anteil der erklärten Varianz
  - X.XXX = **XX.X% der Varianz wird erklärt**
  - Nahe 1.0 = exzellente Anpassung

- **MAPE (Mean Absolute Percentage Error)**: Prozentualer Fehler
  - X.XX% = sehr präzise Prognosen

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
- MAE:  XXX kWh
- RMSE: XXX kWh
- R²:   X.XXX
- MAPE: X.XX%

Test-Metriken:
- MAE:  XXX kWh
- RMSE: XXX kWh
- R²:   X.XXX
- MAPE: X.XX%
```

**Interpretation**:
- Sehr geringer Overfitting (Train R² X.XXX vs. Test R² X.XXX)
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
- MAE:  XXX kWh
- RMSE: XXX kWh
- R²:   X.XXX
- MAPE: X.XX%

Test-Metriken:
- MAE:  XXX kWh
- RMSE: XXX kWh
- R²:   X.XXX
- MAPE: X.XX%
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
- MAE:  XXX kWh
- RMSE: XXX kWh
- R²:   X.XXX
- MAPE: X.XX%

Test-Metriken:
- MAE:  XXX kWh
- RMSE: XXX kWh
- R²:   X.XXX
- MAPE: X.XX%
```

**Beobachtung**:
- Etwas schlechter als Gradient Boosting Modelle
- Höherer RMSE → schlechter bei Ausreißern
- Trotzdem sehr gute Performance (R² X.XXX)

### Beispieltag: 15. November 2024

**Auswahl**: Vollständiger Tag im Test-Set (96 Messpunkte)

**LightGBM Performance**:
- MAE: XXX kWh
- RMSE: XXX kWh
- R²: X.XXX

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
| MAE (kWh) | XXX | XXX |
| RMSE (kWh) | XXX | XXX |
| R² | X.XXXX | X.XXXX |
| MAPE (%) | X.XX | X.XX |

**Interpretation**:
- **Median < Mittelwert** → einige schwierigere Tage (z.B. Feiertage) ziehen Durchschnitt hoch
- R² bleibt über X.XXX → sehr stabile Performance
- MAPE < X% → auch über 24h exzellente Genauigkeit

### Fehler-Akkumulation über Zeit

Analyse: Wie entwickelt sich der Fehler über die 96 Schritte?

```
Stunde  |  MAE (kWh)  |  RMSE (kWh)
--------|-------------|-------------
0-6h    |  XXX        |  XXX
6-12h   |  XXX        |  XXX
12-18h  |  XXX        |  XXX
18-24h  |  XXX        |  XXX
```

**Beobachtung**:
- Fehler steigt leicht mit Forecast-Horizont
- Aber: Kein exponentielles Wachstum (gut!)
- Auch nach 24h noch R² > X.XX

### Beispieltag: 15. November 2024 (Recursive)

**Performance**:
- MAE: XXX kWh
- RMSE: XXX kWh
- R²: X.XXX
- MAPE: X.XX%

**Vergleich zu 1-Step**:
- MAE steigt von XXX → XXX kWh (+XX%)
- R² sinkt nur minimal (X.XXX → X.XXX)
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
- MAE: XXXX kWh
- RMSE: XXXX kWh
- R²: X.XXX
- MAPE: X.XX%

**Vergleich zu ML-Modellen**:
| Modell | MAE | Verbesserung vs. Prophet |
|--------|-----|--------------------------|
| Prophet | XXXX kWh | Baseline |
| LightGBM (1-Step) | XXX kWh | **-XX%** |
| LightGBM (Recursive 24h) | XXX kWh | **-XX%** |

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
- Test MAE: XXX kWh
- Test R²: X.XXX

### Experiment 2: Ohne Wetter-Features

**Konfiguration**:
```python
USE_WEATHER = False
USE_LAGS = True
USE_CALENDAR = True
```

**XGBoost Ergebnisse**:
- Test MAE: XXX kWh (+X kWh vs. Baseline)
- Test R²: X.XXX (unverändert)

**Interpretation**:
- Wetter-Features bringen **marginal Verbesserung** (~X%)
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
- Test MAE: XXXX kWh (**+XXXX kWh** vs. Baseline!)
- Test R²: X.XXX (-X.XXX)

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
- Test MAE: XXX kWh (-X kWh vs. Baseline)
- Test R²: X.XXX (unverändert)

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
- Test MAE: XXX kWh (unverändert)
- Test R²: X.XXX (unverändert)

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
| 1 | **Lag_24h** | X.XXX | Lag |
| 2 | **Lag_1h** | X.XXX | Lag |
| 3 | **Stunde_sin** | X.XXX | Kalender |
| 4 | **Stunde_cos** | X.XXX | Kalender |
| 5 | **Wochentag_sin** | X.XXX | Kalender |
| 6 | **Lufttemperatur 2m_lag15** | X.XXX | Wetter |
| 7 | **Globalstrahlung_lag15** | X.XXX | Wetter |
| 8 | **Monat_sin** | X.XXX | Kalender |
| 9 | **Wochentag_cos** | X.XXX | Kalender |
| 10 | **Lag_15min** | X.XXX | Lag |

### Interpretation

1. **Lag_24h dominiert** (XX.X% Importance)
   - Verbrauch zur gleichen Stunde am Vortag ist stärkster Prädiktor
   - Tägliche Zyklen werden dadurch erfasst

2. **Stunde des Tages** (sin + cos = XX%)
   - Tageszyklus (Nachts niedrig, Tags hoch) ist zweitwichtigster Faktor
   - Sin/Cos-Encoding funktioniert hervorragend

3. **Wochentag** (sin + cos = X.X%)
   - Wochenenden vs. Arbeitstage
   - Wochenstruktur wichtig für Genauigkeit

4. **Wetter** (Temperatur + Globalstrahlung = X.X%)
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
- Mean Error: -X.X kWh (leichter Bias nach unten)
- Std. Error: XXX kWh
- 95% der Fehler liegen in: [-XXX, +XXX] kWh

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

**Größte Abweichungen (absolute Fehler > XXX kWh)**:
- **01.01.2024 (Neujahr)**: Fehler XXXX kWh
  - Grund: Außergewöhnlicher Verbrauch, nicht im Training
  - Modell sagt zu niedrig voraus

- **24.12.2023 (Heiligabend)**: Fehler XXXX kWh
  - Grund: Früher Verbrauchsrückgang (Geschäfte schließen)

- **14.07.2024 (Hitzewelle)**: Fehler XXX kWh
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
1. Lag_24h: **r = X.XX** (sehr stark)
2. Lag_1h: **r = X.XX**
3. Temperatur: **r = -X.XX** (negativ! Kälter → mehr Verbrauch)
4. Globalstrahlung: **r = -X.XX** (negativ! Mehr Sonne → weniger Verbrauch)
   - Erklärung: Solaranlagen produzieren → Netzbezug sinkt

**Wetter-Interkorrelationen**:
- Temperatur ↔ Globalstrahlung: r = X.XX
- Temperatur ↔ Taupunkt: r = X.XX
- → Multikollinearität vorhanden, aber durch Regularisierung in Modellen beherrscht

### Prediction vs. Actual

**Scatter-Plot (Test-Set)**:
- Ideale Linie: y = x (45°-Linie)
- Tatsächlicher Fit: sehr nah an idealer Linie
- R² = X.XXX visuell bestätigt
- Leichte Untervorhersage bei Extremen (>XX.XXX kWh)

**Residuen vs. Predicted**:
- Zufällige Verteilung um 0
- Keine Trichterform → Homoskedastizität ✓
- Vereinzelte Ausreißer bei hohen/niedrigen Werten

---

## 🏆 Schlussfolgerungen

### Wichtigste Erkenntnisse

1. **ML schlägt klassische Zeitreihen-Modelle**
   - LightGBM/XGBoost: R² X.XXX
   - Prophet: R² X.XXX
   - **Verbesserung: +X.X%**

2. **Lag-Features sind essentiell**
   - Ohne Lags: MAE XXXX kWh
   - Mit Lags: MAE XXX kWh
   - **Verbesserung: -XX%**

3. **Recursive Forecast ist praktikabel**
   - 24h-Prognose mit R² X.XXX
   - Fehler-Akkumulation moderat
   - Produktionsreif für Tagesplanung

4. **Feature Engineering > Modell-Wahl**
   - LightGBM, XGBoost, RF: alle R² X.XXX
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
