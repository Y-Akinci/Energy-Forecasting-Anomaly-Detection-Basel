# 📊 Ergebnisse & Experimente

> Detaillierte Modellergebnisse, Experimente und Performance-Analysen für das Energy Forecasting Projekt.

---

## 📑 Inhaltsverzeichnis

1. [Übersicht](#übersicht)
2. [1-Step Forecast Ergebnisse](#1-step-forecast-ergebnisse)
3. [Multi-Output 24h Global](#multi-output-24h-global)
4. [Multi-Output 24h Block](#multi-output-24h-block)
5. [Recursive 24h Forecast](#recursive-24h-forecast)
6. [Feature Importance](#feature-importance)

---

## 🎯 Übersicht

### Modellperformance (LightGBM)

| Ansatz | Datensatz | MAE (kWh) | RMSE (kWh) | R² | MAPE (%) |
|--------|-----------|-----------|------------|-----|----------|
| **1-Step (15min)** | **Train** | **298.15** | **394.22** | **0.9971** | **0.86%** |
| **1-Step (15min)** | **Test** | **412.54** | **564.30** | **0.9935** | **1.19%** |
| Multi-Output 24h Global | Train | 568.11 | 744.87 | 0.9897 | 1.63% |
| Multi-Output 24h Global | Test | 1069.02 | 1551.08 | 0.9510 | 3.10% |
| Recursive 24h (Mean) | Train | 516.30 | 660.73 | 0.9575 | 1.51% |
| Recursive 24h (Mean) | Test | 858.15 | 1073.77 | 0.8632 | 2.48% |

### Metriken-Erklärung

- **MAE (Mean Absolute Error)**: Durchschnittlicher absoluter Fehler in kWh
  - Interpretation: Im Test liegt die 1-Step Vorhersage ±413 kWh daneben
  - Bei ~35.000 kWh Durchschnittsverbrauch = **1.19% Fehler**

- **RMSE (Root Mean Squared Error)**: Wurzel der mittleren quadratischen Abweichung
  - Bestraft große Fehler stärker als MAE
  - Wichtig für Ausreißer-Erkennung

- **R² (Bestimmtheitsmaß)**: Anteil der erklärten Varianz
  - 0.9935 = **99.35% der Varianz wird erklärt**
  - Nahe 1.0 = exzellente Anpassung

- **MAPE (Mean Absolute Percentage Error)**: Prozentualer Fehler
  - 1.19% (1-Step Test) = sehr präzise Prognosen

---

## 🔹 1-Step Forecast Ergebnisse

### LightGBM Modell

**Konfiguration**:
- Datenzeitraum: 2020-08-31 bis 2024-12-31
- Train/Test-Split: 70% / 30% (chronologisch)
- Features: Kalender (sin/cos) + Lags + Wetter-Lags
- Modell: LightGBM

#### Performance-Metriken

```
Train-Metriken:
- MAE:  298.15 kWh
- RMSE: 394.22 kWh
- R²:   0.9971
- MAPE: 0.86%

Test-Metriken:
- MAE:  412.54 kWh
- RMSE: 564.30 kWh
- R²:   0.9935
- MAPE: 1.19%
```

**Interpretation**:
- Sehr geringer Overfitting (Train R² 0.9971 vs. Test R² 0.9935)
- Robuste Performance auf ungesehenen Daten
- MAPE 1.19% → hervorragende Genauigkeit
- Bei ~35.000 kWh Durchschnittsverbrauch liegt der durchschnittliche Fehler bei nur 413 kWh

---

## 📊 Multi-Output 24h Global

### Konzept

- **Ansatz**: Der gesamte 24-Stunden-Horizont (96 Zeitpunkte) wird gleichzeitig als Vektor vorhergesagt
- **Modell**: LightGBM mit Multi-Output Regression
- **Vorteil**: Alle Zeitpunkte werden simultan optimiert
- **Nachteil**: Komplexeres Modell, höhere Fehlerraten

### Performance-Metriken

```
Train-Metriken:
- MAE:  568.11 kWh
- RMSE: 744.87 kWh
- R²:   0.9897
- MAPE: 1.63%

Test-Metriken:
- MAE:  1069.02 kWh
- RMSE: 1551.08 kWh
- R²:   0.9510
- MAPE: 3.10%
```

**Interpretation**:
- Deutlich höherer Fehler als 1-Step (MAE 1069 kWh vs. 413 kWh)
- R² bleibt dennoch hoch bei 0.9510
- MAPE 3.10% ist akzeptabel, aber nicht optimal für Produktionsumgebung

---

## 📈 Multi-Output 24h Block

### Konzept

- **Ansatz**: Tägliche Blöcke (Start 00:00 lokal) werden einzeln evaluiert
- **Auswertung**: Durchschnittliche Performance über alle Tage

### Aggregierte Ergebnisse

**Train (1108 Tage):**
```
- MAE (Mean):   701.25 kWh
- MAE (Median): 649.74 kWh
- RMSE (Mean):  866.82 kWh
- R² (Mean):    0.9649
- MAPE (Mean):  2.03%
```

**Test (475 Tage):**
```
- MAE (Mean):   1405.26 kWh
- MAE (Median): 1234.68 kWh
- RMSE (Mean):  1723.25 kWh
- R² (Mean):    0.7809
- MAPE (Mean):  4.07%
```

**Interpretation**:
- Median < Mittelwert → einige schwierigere Tage ziehen Durchschnitt hoch
- R² von 0.7809 zeigt moderate Generalisierung
- MAPE 4.07% ist für 24h-Prognosen noch akzeptabel

---

## 🔁 Recursive 24h Forecast ⭐ Empfohlen

### Konzept

- **Start**: 00:00 Uhr (lokal)
- **Methode**: Iteratives 1-Step-Forecasting (96× wiederholt)
- **Besonderheit**: Jede Vorhersage wird zur Basis für die nächste
- **Vorteil**: Beste Generalisierung im Test-Set

### Aggregierte Ergebnisse (Tägliche Blöcke)

**Train (1108 Tage):**
```
- MAE (Mean):   516.30 kWh
- MAE (Median): 451.20 kWh
- RMSE (Mean):  660.73 kWh
- R² (Mean):    0.9575
- MAPE (Mean):  1.51%
```

**Test (474 Tage):**
```
- MAE (Mean):   858.15 kWh
- MAE (Median): 677.15 kWh
- RMSE (Mean):  1073.77 kWh
- R² (Mean):    0.8632
- MAPE (Mean):  2.48%
```

**Interpretation**:
- **Deutlich besser als Multi-Output Block** (MAPE 2.48% vs. 4.07%)
- Median < Mittelwert → einige schwierigere Tage (z.B. Feiertage) ziehen Durchschnitt hoch
- R² von 0.8632 → sehr stabile Performance auch über 24h
- MAPE 2.48% → auch über volle Tagesprognose exzellente Genauigkeit

### Warum ist Recursive besser als Multi-Output?

1. **Zeitliche Abhängigkeit**: Das Modell nutzt die vorherigen Vorhersagen als Input
2. **Geringere Fehlerfortpflanzung**: Trotz rekursiver Natur akkumuliert der Fehler moderat
3. **Bessere Generalisierung**: R² 0.8632 vs. 0.7809 bei Multi-Output Block
4. **Produktionsreif**: Mit MAPE 2.48% ideal für Tagesplanung

---

## 🎯 Feature Importance

### Wichtigste Feature-Gruppen (LightGBM)

Basierend auf der Modellanalyse sind folgende Feature-Gruppen essentiell:

| Feature-Gruppe | Bedeutung | Beschreibung |
|----------------|-----------|--------------|
| **Lags (Verbrauch)** | ⭐⭐⭐⭐⭐ | Historischer Verbrauch (15min, 1h, 24h zurück) - stärkster Prädiktor |
| **Kalender (sin/cos)** | ⭐⭐⭐⭐ | Stunde, Wochentag, Monat - erfasst zeitliche Muster |
| **Wetter-Lags** | ⭐⭐⭐ | Temperatur, Globalstrahlung - relevant bei Extremwetter |

### Interpretation

1. **Lag-Features dominieren**
   - Verbrauch zur gleichen Stunde am Vortag ist stärkster Prädiktor
   - Tägliche Zyklen werden dadurch erfasst
   - Autokorrelation ist Schlüssel zum Erfolg

2. **Stunde des Tages** (sin/cos)
   - Tageszyklus (Nachts niedrig, Tags hoch)
   - Sin/Cos-Encoding funktioniert hervorragend für zyklische Features

3. **Wochentag** (sin/cos)
   - Wochenenden vs. Arbeitstage
   - Wochenstruktur wichtig für Genauigkeit

4. **Wetter**
   - Temperatur und Globalstrahlung haben messbaren Einfluss
   - Besonders relevant bei extremen Bedingungen
   - Sommer: Kühlbedarf (Klimaanlagen)
   - Winter: Heizbedarf (Elektroheizungen)

---

## 🏆 Schlussfolgerungen

### Wichtigste Erkenntnisse

1. **LightGBM liefert exzellente Ergebnisse**
   - 1-Step Forecast: R² 0.9935, MAPE 1.19%
   - Sehr geringe Overfitting-Tendenz
   - Robuste Performance auf ungesehenen Daten

2. **Recursive Forecast übertrifft Multi-Output**
   - Recursive: MAPE 2.48% (Test)
   - Multi-Output Block: MAPE 4.07% (Test)
   - **Verbesserung: -39%** beim MAPE
   - Trotz rekursiver Natur moderate Fehler-Akkumulation

3. **Verschiedene Forecast-Strategien im Vergleich**
   - **1-Step (15min)**: Beste Genauigkeit (MAPE 1.19%), ideal für kurzfristige Prognosen
   - **Recursive 24h**: Beste 24h-Prognose (MAPE 2.48%), empfohlen für Tagesplanung
   - **Multi-Output Global**: Moderate Performance (MAPE 3.10%), schnellere Berechnung
   - **Multi-Output Block**: Schwächste Generalisierung (MAPE 4.07%)

4. **Feature Engineering ist essentiell**
   - Lag-Features (Verbrauch der letzten Stunden/Tage) sind der stärkste Prädiktor
   - Sin/Cos-Encoding für zyklische Features funktioniert hervorragend
   - Wetter-Features bringen zusätzliche Genauigkeit bei Extrembedingungen

5. **Produktionsreife erreicht**
   - R² von 0.8632 für 24h-Prognosen
   - MAPE von 2.48% ermöglicht zuverlässige Tagesplanung
   - Modell ist stabil und generalisiert gut

### Empfehlungen

**Für Produktion**:
- **Modell**: LightGBM
- **Forecast-Strategie**: Recursive 24h (beste Balance aus Genauigkeit und Praktikabilität)
- **Features**: Lags + Kalender (sin/cos) + Wetter-Lags
- **Retraining**: Regelmäßig mit neuen Daten (monatlich empfohlen)

**Für weitere Verbesserungen**:
1. Holiday-Encoding (Schweizer Feiertage, Basel-spezifische Events) hinzufügen
2. Extreme-Weather-Flags für außergewöhnliche Wetterbedingungen
3. Separate Modelle für verschiedene Jahreszeiten testen
4. Ensemble-Ansätze evaluieren

### Vergleich der Prognose-Strategien

| Strategie | Test MAPE | Test R² | Anwendungsfall |
|-----------|-----------|---------|----------------|
| **1-Step** | **1.19%** | **0.9935** | Kurzfristprognosen (15min) |
| **Recursive 24h** ⭐ | **2.48%** | **0.8632** | Tagesplanung (empfohlen) |
| **Multi-Output Global** | **3.10%** | **0.9510** | Schnelle 24h-Prognosen |
| **Multi-Output Block** | **4.07%** | **0.7809** | Experimentell |

---

**← Zurück zu [Technical-Details.md](Technical-Details.md) | Weiter zu [Data-Pipeline.md](Data-Pipeline.md) →**
