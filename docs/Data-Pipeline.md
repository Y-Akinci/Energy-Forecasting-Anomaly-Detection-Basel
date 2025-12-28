# 🔄 Data Pipeline

> Detaillierte Beschreibung der Datenaufbereitung, Feature Engineering und Pipeline-Architektur.

---

## 📑 Inhaltsverzeichnis

1. [Pipeline-Übersicht](#pipeline-übersicht)
2. [Datenquellen](#datenquellen)
3. [Data Collection](#data-collection)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Feature Selection](#feature-selection)
7. [Datenexport](#datenexport)
8. [Herausforderungen & Lösungen](#herausforderungen--lösungen)

---

## 🏗️ Pipeline-Übersicht

```
┌─────────────────┐     ┌─────────────────┐
│ Stromverbrauch  │     │  Wetterdaten    │
│   (CSV, lokal)  │     │  (CSV, UTC)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌────────────────────────────────────────┐
│        1. Data Collection              │
│  • Laden & Parsing                     │
│  • Zeitstempel-Konvertierung (→ UTC)   │
│  • Index-Erstellung (15min-Achse)      │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│        2. Data Merging                 │
│  • LEFT Join auf Strom-Index           │
│  • Wetter: 10min → 15min (Interpolation)│
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│        3. Data Preprocessing           │
│  • Duplikate entfernen                 │
│  • Fehlende Werte behandeln            │
│  • Datentypen konvertieren             │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│        4. Feature Engineering          │
│  • Zeitfeatures (lokal + sin/cos)      │
│  • Lag-Features (15min, 1h, 24h)       │
│  • Wetter-Lags (15min verzögert)       │
│  • Differenz-Features                  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│        5. Feature Selection            │
│  • Redundante Features entfernen       │
│  • Original-Wetter löschen (nur Lags)  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│        6. Export                       │
│  • processed_merged_features.csv       │
│  • Shape: ~480.000 × 60+               │
└────────────────────────────────────────┘
```

**Pipeline-Datei**: `Data_Preparation/data_preparation.py`

---

## 📊 Datenquellen

### Stromverbrauchsdaten

**Datei**: `251006_StromverbrauchBasel2012-2025.csv`

| Eigenschaft | Wert |
|-------------|------|
| **Zeitraum** | 01.01.2012 - 29.09.2025 |
| **Intervall** | 15 Minuten |
| **Zeilen** | 481.959 |
| **Zeitzone** | Europe/Zurich (lokal) |
| **Quelle** | [OpenData Basel-Stadt](https://opendata.swiss/de/dataset/kantonaler-stromverbrauch-netzlast) |

**Spalten**:
- `Start der Messung (Text)`: Zeitstempel (String, lokal)
- `Stromverbrauch`: kWh im 15min-Intervall
- `Grundversorgte Kunden`: Anzahl Kunden (IWB-Vertrag)
- `Freie Kunden`: Anzahl Kunden (Markt-Vertrag)
- `Jahr`, `Monat`, `Tag`, `Wochentag`, `Tag des Jahres`, `Quartal`, `Woche des Jahres`

**Datenqualität**:
- ✅ Keine Duplikate
- ⚠️ 52 fehlende Zeitpunkte (Sommerzeit-Umstellung)
- ⚠️ ~62% fehlende Werte bei Kundenzahlen (vor 2020)

### Wetterdaten

**Dateien**:
- `251107_Wetterdaten_Basel_2010-2019.csv`
- `251107_Wetterdaten_Basel_2020-2024.csv`

| Eigenschaft | Wert |
|-------------|------|
| **Zeitraum** | 01.01.2010 - 31.12.2024 |
| **Intervall** | 10 Minuten |
| **Zeilen** | 788.977 (kombiniert) |
| **Zeitzone** | UTC |
| **Quelle** | [MeteoSchweiz - Station BAS](https://www.meteoschweiz.admin.ch/) |

**Wichtigste Spalten**:
- **Temperatur**: `Lufttemperatur 2 m ü. Boden`, `Taupunkt`, `Chilltemperatur`
- **Strahlung**: `Globalstrahlung`, `Diffusstrahlung`, `Langwellige Einstrahlung`, `Sonnenscheindauer`
- **Wind**: `Windgeschwindigkeit skalar`, `Böenspitze`, `Windrichtung`
- **Feuchtigkeit**: `relative Luftfeuchtigkeit`, `Dampfdruck`
- **Druck**: `Luftdruck auf Barometerhöhe`
- **Niederschlag**: `Niederschlag; Zehnminutensumme`

**Datenqualität**:
- ✅ < 0.5% fehlende Werte
- ✅ Konsistente 10min-Intervalle

---

## 📥 Data Collection

### 1. Stromverbrauch laden

```python
# data_preparation.py, Zeilen 15-20

# STROM laden (helpers -> Europe/Zurich) und auf UTC bringen
power_local = helpers.csv_file()      # tz-aware Europe/Zurich
power = power_local.tz_convert("UTC")

# Exakte 15-min Achse auf Basis der Stromdaten (LEFT-Master-Achse)
full_idx_utc = pd.date_range(power.index.min(), power.index.max(), freq="15min", tz="UTC")
power_15 = power.reindex(full_idx_utc)
```

**Was passiert?**
1. **Helper lädt Daten** mit korrekter Zeitzone (`Europe/Zurich`)
2. **Konvertierung zu UTC** → keine Zeitsprünge durch Sommerzeit
3. **Master-Index erstellen** (15min-Raster, UTC)
4. **Reindex** → fehlende Zeitpunkte bekommen `NaN`

### 2. Wetterdaten laden & mergen

```python
# data_preparation.py, Zeilen 22-50

f1 = os.path.join(DATA_DIR, "raw data", "251107_Wetterdaten_Basel_2010-2019.csv")
f2 = os.path.join(DATA_DIR, "raw data", "251107_Wetterdaten_Basel_2020-2024.csv")

w1 = pd.read_csv(f1, sep=";", decimal=",", encoding="latin1", low_memory=False)
w2 = pd.read_csv(f2, sep=";", decimal=",", encoding="latin1", low_memory=False)

# Zeitstempel parsen (direkt als UTC)
w1["DateTime"] = pd.to_datetime(w1["Date and Time"], dayfirst=True, utc=True)
w2["DateTime"] = pd.to_datetime(w2["Date and Time"], dayfirst=True, utc=True)

# Zusammenführen
weather = pd.concat([w1, w2], axis=0)
weather = weather[~weather.index.duplicated(keep="last")].sort_index()
```

**Besonderheiten**:
- `sep=";"` → CSV mit Semikolon
- `decimal=","` → Dezimalkomma (europäisches Format)
- `encoding="latin1"` → Umlaute (ä, ö, ü) korrekt
- `dayfirst=True` → DD.MM.YYYY Format

### 3. Interpolation: 10min → 15min

```python
# data_preparation.py, Zeilen 52-74

# Numerische Spalten: zeitbasierte Interpolation exakt auf die 15-min Strom-Achse
num_cols = weather.select_dtypes(include="number").columns
num_15 = weather[num_cols].reindex(full_idx_utc).interpolate(method="time", limit_direction="both")

# Nicht-numerische Spalten: nächster Wert (nearest) ohne Verschiebung
other_cols = weather.columns.difference(num_cols)
if len(other_cols) > 0:
    other_15 = pd.merge_asof(
        target,
        weather[other_cols].sort_index(),
        left_index=True, right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("7min30s")
    )
    weather_15 = pd.concat([num_15, other_15], axis=1)
```

**Warum `method="time"`?**
- **Lineare Interpolation** basierend auf Zeit-Distanz
- Beispiel: 10:00 (20°C), 10:10 (22°C) → 10:15 ≈ 22.33°C
- Physikalisch sinnvoll für Temperatur, Strahlung, etc.

**Warum `tolerance="7min30s"`?**
- Max. Distanz zwischen 10min-Stützstellen: 5min
- +2.5min Puffer → 7.5min
- Verhindert falsche Zuordnung bei Lücken

### 4. LEFT Join

```python
# data_preparation.py, Zeile 77

merged_15 = power_15.join(weather_15, how="left")
```

**LEFT Join bedeutet**:
- **Strom-Zeitachse** ist Master (alle 15min-Punkte bleiben)
- **Wetter wird hinzugefügt** (wenn verfügbar)
- Fehlende Wetter-Daten → `NaN`

---

## 🧹 Data Preprocessing

### 1. Data Cleaning

```python
# data_preparation.py, Zeile 88

merged_15.dropna(subset="Stromverbrauch", inplace=True)
```

**Entfernt**:
- 52 Zeitpunkte (Sommerzeit-Umstellung)
- Evtl. weitere Lücken in Stromverbrauchsdaten

### 2. Duplikate entfernen

```python
# data_preparation.py, Zeile 50

weather = weather[~weather.index.duplicated(keep="last")].sort_index()
```

**Strategie**: Bei doppelten Timestamps → behalte **letzten** Eintrag

### 3. Datentypen konvertieren

#### Integers

```python
# data_preparation.py, Zeilen 92-100

int_cols = [
    "Jahr", "Monat", "Tag", "Wochentag", "Tag des Jahres",
    "Quartal", "Woche des Jahres",
]

for col in int_cols:
    merged_15[col] = pd.to_numeric(
        merged_15[col], errors="coerce"
    ).astype("Int64")  # ← Int64 unterstützt NaN!
```

**Warum `Int64` statt `int64`?**
- `int64` kann keine `NaN` darstellen
- `Int64` (pandas nullable integer) erlaubt fehlende Werte

#### Floats

```python
# data_preparation.py, Zeilen 103-133

float_cols = [
    'Stromverbrauch',
    'Grundversorgte Kunden',
    'Freie Kunden',
    'Globalstrahlung; Zehnminutenmittel',
    # ... alle Wetter-Spalten
]

for col in float_cols:
    merged_15[col] = pd.to_numeric(merged_15[col], errors="coerce").astype("float64")
```

#### Kategorische Variablen

```python
# data_preparation.py, Zeilen 136-145

categorical_features = [
    "Jahr",
    "Monat",
    "Wochentag",
    "Quartal",
    "Woche des Jahres"
]

for col in categorical_features:
    merged_15[col] = merged_15[col].astype("category")
```

**Warum `category`?**
- Memory-Optimierung (speichert nur unique Werte + Mapping)
- Schnellere Gruppierungen
- Explizite Kennzeichnung diskreter Werte

---

## 🛠️ Feature Engineering

### 1. Lokale Zeitfeatures

```python
# data_preparation.py, Zeilen 150-163

# Lokale Zeit aus UTC-Index erzeugen
idx_local = merged_15.index.tz_convert("Europe/Zurich")

# Lokale Datum- und Zeitspalten erstellen
merged_15["Datum (Lokal)"] = idx_local.date
merged_15["Zeit (Lokal)"] = idx_local.time
merged_15["Stunde (Lokal)"] = idx_local.hour
```

**Warum lokal?**
- Stunde `23:00 UTC` ≠ `23:00 Lokal` (wichtig für Tagesmuster!)
- Lokale Zeit = Nutzerverhalten (Arbeit, Freizeit, Schlaf)

### 2. Binäre Zeitfeatures

```python
# data_preparation.py, Zeilen 177-181

wd = merged_15["Wochentag"].astype("int")

merged_15["IstSonntag"] = (wd == 6).astype(int)
merged_15["IstArbeitstag"] = (wd < 5).astype(int)
```

**Logik**:
- **Sonntag**: Wochentag == 6 (Pandas: 0=Montag, 6=Sonntag)
- **Arbeitstag**: Montag-Freitag (Wochentag 0-4)
- **Samstag**: Weder Sonntag noch Arbeitstag → implizit kodiert

### 3. Lag-Features

#### Stromverbrauchs-Lags

```python
# data_preparation.py, Zeilen 183-189

merged_15["Lag_15min"] = merged_15["Stromverbrauch"].shift(1)
merged_15["Lag_30min"] = merged_15["Stromverbrauch"].shift(2)
merged_15["Lag_1h"] = merged_15["Stromverbrauch"].shift(4)   # 4 × 15min = 1h
merged_15["Lag_24h"] = merged_15["Stromverbrauch"].shift(96)  # 96 × 15min = 24h
merged_15["Grundversorgte Kunden_Lag_15min"] = merged_15["Grundversorgte Kunden"].shift(1)
merged_15["Freie Kunden_Lag_15min"] = merged_15["Freie Kunden"].shift(1)
```

**Shift-Logik**:
| Feature | Shift | Bedeutung |
|---------|-------|-----------|
| Lag_15min | 1 | Verbrauch vor 15min |
| Lag_30min | 2 | Verbrauch vor 30min |
| Lag_1h | 4 | Verbrauch vor 1h |
| Lag_24h | 96 | Verbrauch vor 24h (gleiche Stunde gestern) |

**Warum 96 für 24h?**
```
1 Stunde = 60min ÷ 15min = 4 Intervalle
24 Stunden = 24 × 4 = 96 Intervalle
```

#### Wetter-Lags

```python
# data_preparation.py, Zeilen 223-226

for col in weather_cols:
    if col in merged_15.columns:
        merged_15[f"{col}_lag15"] = merged_15[col].shift(1)
```

**Kritisch!** Wetter wird **15min verzögert** → kein Data Leakage!

Beispiel:
```
Zeitpunkt: 12:00
Temperatur_lag15: Temperatur von 11:45
```

Beim Forecasting für 12:00 kennt man nur Daten bis 11:45!

### 4. Differenz-Feature

```python
# data_preparation.py, Zeile 261

merged_15["Diff_15min"] = merged_15["Lag_15min"] - merged_15["Lag_30min"]
```

**Interpretation**:
- **Positiv**: Verbrauch steigt
- **Negativ**: Verbrauch sinkt
- **Nahe 0**: Stabiler Verbrauch

Hilft Modell, **Trends** zu erkennen!

### 5. NaN-Handling (Lags)

```python
# data_preparation.py, Zeilen 191-193

lag_cols = ["Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h"]
merged_15.dropna(subset=lag_cols, inplace=True)
```

**Problem**: Erste 96 Zeilen haben keine `Lag_24h` (nichts zum Shiften)

**Lösung**: Entfernen → Training startet erst, wenn alle Lags verfügbar

---

## ✂️ Feature Selection

### 1. Original-Wetter entfernen

```python
# data_preparation.py, Zeilen 228-258

original_weather_cols = [
    'Globalstrahlung; Zehnminutenmittel',
    'Diffusstrahlung; Zehnminutenmittel',
    # ... alle ursprünglichen Wetter-Features
]

merged_15.drop(columns=[col for col in original_weather_cols if col in merged_15.columns],
               inplace=True)
```

**Warum?**
- Original-Wetter = Data Leakage (Zukunftsinformation)
- Nur `_lag15`-Varianten bleiben → sicher verwendbar

### 2. Redundante Features entfernen

```python
# data_preparation.py, Zeilen 270-273

delete_columns = ["Jahr", "Date", "Time", "Date and Time", "Start der Messung (Text)",
                  "station_abbr", "Grundversorgte Kunden", "Freie Kunden"]
cols_to_drop = [c for c in delete_columns if c in merged_15.columns]
merged_15.drop(columns=cols_to_drop, inplace=True)
```

**Entfernte Features**:
- **Jahr**: Redundant zu `Monat` + `Tag des Jahres`
- **Grundversorgte/Freie Kunden**: Schwache Prädiktoren (nur Lag-Versionen bleiben)
- **Text-Zeitstempel**: Redundant zu Index
- **station_abbr**: Konstant (nur BAS)

**Resultat**: ~60 Features (von ursprünglich 100+)

---

## 💾 Datenexport

```python
# data_preparation.py, Zeilen 278-286

output_path = os.path.join(DATA_DIR, "processed_merged_features.csv")

merged_15.to_csv(output_path, sep=";", encoding="latin1")

print("Export erfolgreich!")
print("Datei gespeichert unter:", output_path)
print("Shape:", merged_15.shape)
```

**Ausgabe**:
```
Export erfolgreich!
Datei gespeichert unter: data/processed_merged_features.csv
Shape: (478834, 62)
```

**Format**:
- **Separator**: `;` (europäischer Standard)
- **Encoding**: `latin1` (für Umlaute)
- **Index**: `Start der Messung (UTC)` (DateTime, UTC)
- **Spalten**: 62 Features + Target

---

## 🚧 Herausforderungen & Lösungen

### Challenge 1: Zeitstempel-Chaos

**Problem**:
- Stromverbrauch: Lokal (Europe/Zurich)
- Wetterdaten: UTC
- Sommerzeit-Umstellung: Zeitsprünge

**Lösung**:
1. Alles auf UTC konvertieren
2. Master-Index in UTC erstellen
3. Features in lokaler Zeit **berechnen**, aber Index bleibt UTC
4. Keine Zeitsprünge mehr!

### Challenge 2: Interpolation 10min → 15min

**Problem**:
- Wetter: 10min-Intervalle (6× pro Stunde)
- Strom: 15min-Intervalle (4× pro Stunde)
- Nicht ganzzahlig teilbar!

**Lösung**:
- `method="time"` → zeitgewichtete lineare Interpolation
- Beispiel: 10:00 (20°C), 10:10 (22°C) → 10:15 = 20 + (22-20) × (15/10) = 23°C

### Challenge 3: Data Leakage mit Wetter

**Problem**:
- Temperatur um 12:00 zur Vorhersage von Verbrauch um 12:00 verwenden?
- Unrealistisch! Man kennt Wetter nur aus der Vergangenheit

**Lösung**:
- Alle Wetter-Features als `_lag15` (15min verzögert)
- Original-Features löschen
- Modell sieht nur Vergangenheit

### Challenge 4: Sommerzeit-Lücken

**Problem**:
- März: 02:00-03:00 existiert nicht (4 Werte fehlen)
- Oktober: 02:00-03:00 doppelt (4 Werte verworfen)
- 52 fehlende Zeitpunkte über alle Jahre

**Lösung**:
- `dropna(subset="Stromverbrauch")` → entfernt sie
- Keine Interpolation! (physikalisch nicht gemessen)
- Modelle können mit kleinen Lücken umgehen

### Challenge 5: Multikollinearität

**Problem**:
- Windgeschwindigkeit in km/h **und** m/s
- Luftdruck QFF, QNH, Barometer
- Temperatur 2m, Bodenoberfläche, Gras

**Lösung**:
- Feature-Reduktion (nur eine Variante behalten)
- `EXCLUDE_WEATHER`-Liste in Modellierungsskripten
- Keine Performance-Einbuße!

### Challenge 6: Kategorische vs. Numerische Features

**Problem**:
- Monat 12 (Dezember) und Monat 1 (Januar) sind sich ähnlich
- Numerisch: 1, 2, ..., 12 → falsche Ordnung
- One-Hot: 12 Spalten → zu viele Dimensionen

**Lösung**:
- **Sin/Cos-Encoding** (siehe Technical-Details.md)
- Monat_sin + Monat_cos → zyklische Natur erfasst
- Nur 2 Features statt 12!

---

## 📊 Pipeline-Metriken

### Datenmenge

| Phase | Zeilen | Spalten |
|-------|--------|---------|
| **Stromverbrauch (roh)** | 481.959 | 12 |
| **Wetterdaten (roh)** | 788.977 | 27 |
| **Nach Merge** | 481.959 | 39 |
| **Nach Feature Engineering** | 481.863 | 102 |
| **Nach Feature Selection** | 478.834 | 62 |
| **Final (processed)** | 478.834 | 62 |

### Verarbeitungszeit

| Schritt | Dauer (ca.) |
|---------|-------------|
| Daten laden | 5s |
| Interpolation | 15s |
| Merge | 2s |
| Feature Engineering | 8s |
| Export | 10s |
| **Gesamt** | **~40s** |

(Auf Standard-Laptop, Intel i7, 16GB RAM)

### Speicherverbrauch

| Datei | Größe |
|-------|-------|
| Stromverbrauch (CSV) | ~45 MB |
| Wetterdaten (CSV, beide) | ~120 MB |
| Processed (CSV) | ~65 MB |

---

## 🎓 Best Practices

### 1. Zeitstempel-Handling

✅ **DO**:
- Konsistente Zeitzone verwenden (UTC für Index)
- Lokale Zeit nur für Features berechnen
- `tz-aware` DataFrames verwenden

❌ **DON'T**:
- `tz-naive` Timestamps mischen mit `tz-aware`
- String-Timestamps in Berechnungen verwenden
- Sommerzeit ignorieren

### 2. Feature Engineering

✅ **DO**:
- Lag-Features für Zeitreihen
- Sin/Cos für zyklische Features
- Domain Knowledge nutzen (z.B. Wetter-Einfluss)

❌ **DON'T**:
- Zukunftsinformationen verwenden (Data Leakage!)
- Hochkorrelierte Features behalten (Multikollinearität)
- Features ohne Interpretierbarkeit

### 3. Data Validation

✅ **DO**:
- Duplikate prüfen (`index.duplicated()`)
- Fehlende Werte quantifizieren (`isna().sum()`)
- Datentypen validieren (`dtypes`)
- Wertebereich prüfen (min/max plausibel?)

❌ **DON'T**:
- Blind `fillna(0)` verwenden
- Outliers ohne Analyse entfernen
- Fehler stillschweigend ignorieren

### 4. Performance

✅ **DO**:
- `category` dtype für diskrete Werte
- Vectorized operations (keine Loops)
- Chained operations vermeiden (`df.copy()` nutzen)

❌ **DON'T**:
- `apply()` wo `.shift()`, `.rolling()` möglich
- Unnötige Kopien (`df.copy()` in jedem Schritt)
- Ganze DataFrames in Memory bei Streaming-Verarbeitung

---

## 🔍 Debugging-Tipps

### Zeitstempel-Probleme debuggen

```python
# Zeitzone prüfen
print("Index tz:", merged_15.index.tz)

# Duplikate finden
print("Duplikate:", merged_15.index.duplicated().sum())

# Lücken identifizieren
full_idx = pd.date_range(merged_15.index.min(), merged_15.index.max(), freq="15min", tz="UTC")
missing = full_idx.difference(merged_15.index)
print(f"Fehlende Slots: {len(missing)}")
print(missing[:10])  # Erste 10 anzeigen
```

### Feature-Validierung

```python
# NaNs pro Feature
print(merged_15.isna().sum().sort_values(ascending=False).head(10))

# Wertebereich
print(merged_15[['Stromverbrauch', 'Lag_24h']].describe())

# Korrelations-Check
print(merged_15[['Lag_15min', 'Lag_30min', 'Lag_1h']].corr())
```

### Memory-Profiling

```python
# Memory Usage
print(merged_15.memory_usage(deep=True).sum() / 1024**2, "MB")

# Per dtype
print(merged_15.memory_usage(deep=True).groupby(merged_15.dtypes).sum() / 1024**2)
```

---

## 📚 Weiterführende Ressourcen

- [Pandas Time Series Docs](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Time Zone Handling in Python](https://realpython.com/python-datetime/)
- [Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)

---

**← Zurück zu [Results.md](Results.md) | Zurück zum [README](../README.md) →**
