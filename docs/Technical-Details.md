# 🔧 Technical Details

> Technische Implementierungsdetails und Erklärungen zu spannenden Code-Stellen im Energy Forecasting Projekt.

---

## 📑 Inhaltsverzeichnis

1. [Zeitstempel-Management](#zeitstempel-management)
2. [Rekursiver 24h-Forecast](#rekursiver-24h-forecast)
3. [Feature Engineering](#feature-engineering)
4. [Data Leakage Prevention](#data-leakage-prevention)
5. [Sin/Cos-Encoding für zyklische Features](#sincos-encoding-für-zyklische-features)
6. [Pipeline-Architektur](#pipeline-architektur)

---

## ⏰ Zeitstempel-Management

### Problem: UTC vs. Lokalzeit

Eine der größten Herausforderungen war das korrekte Handling von Zeitstempeln:

- **Stromverbrauchsdaten**: Gemessen in lokaler Zeit (`Europe/Zurich`)
- **Wetterdaten**: Bereitgestellt in UTC
- **Zeitumstellung**: Sommer-/Winterzeit führt zu fehlenden/doppelten Werten

### Lösung: Konsistente UTC-Basis

```python
# data_preparation.py, Zeilen 15-21

# STROM laden (helpers -> Europe/Zurich) und auf UTC bringen
power_local = helpers.csv_file()      # tz-aware Europe/Zurich
power = power_local.tz_convert("UTC")

# Exakte 15-min Achse auf Basis der Stromdaten (LEFT-Master-Achse)
full_idx_utc = pd.date_range(power.index.min(), power.index.max(), freq="15min", tz="UTC")
power_15 = power.reindex(full_idx_utc)
```

**Warum ist das spannend?**

1. **Alle Daten werden auf UTC konvertiert** → keine Zeitsprünge durch Sommerzeit
2. **Master-Index**: Stromverbrauch definiert die 15min-Achse (LEFT-Join)
3. **Wetterdaten werden interpoliert** auf diese Achse (siehe nächster Abschnitt)

### Sommerzeit-Problematik

**März (Uhr wird vorgestellt)**:
- Die Stunde 02:00-03:00 existiert nicht
- → 4 fehlende 15-Minuten-Werte

**Oktober (Uhr wird zurückgestellt)**:
- Die Stunde 02:00-03:00 würde doppelt vorkommen
- → Eine Stunde wird verworfen → 4 fehlende Werte

**Ergebnis**: Über alle Jahre fehlen exakt **52 Zeitpunkte** (2 Tage × 13 Jahre × 2 Umstellungen)

**Lösung im Code**:
```python
# data_preparation.py, Zeile 88
merged_15.dropna(subset="Stromverbrauch", inplace=True)
```

Diese fehlenden Werte werden einfach entfernt, da sie real nie gemessen wurden.

---

## 🔁 Rekursiver 24h-Forecast

### Konzept

Der rekursive Forecast ist der **empfohlene Ansatz** für 24h-Prognosen:

- Startet mit Historie (z.B. letzte 7 Tage)
- Sagt 15min voraus (1-Step)
- Fügt Vorhersage zur Historie hinzu
- **Wiederholt 96× für 24h** (24h × 4 = 96 Intervalle)

### Implementation

```python
# multistep_forecast_recursive.py, Zeilen 192-270

def forecast_multistep(pipe, df_history, df_full, steps=96):
    df_temp = df_history.copy().sort_index()
    preds = []

    for _ in range(steps):
        last_ts = df_temp.index[-1]
        next_ts = last_ts + pd.Timedelta("15min")

        row = {}

        # 1. Kalender-Features berechnen
        if "Monat_sin" in FEATURES or "Monat_cos" in FEATURES:
            s, c = _sin_cos(next_ts.month, 12)
            row["Monat_sin"] = s
            row["Monat_cos"] = c

        # 2. Lag-Features aus vorherigen Vorhersagen
        if "Lag_15min" in FEATURES:
            row["Lag_15min"] = df_temp.iloc[-1][TARGET_COL]
        if "Lag_1h" in FEATURES and len(df_temp) >= 4:
            row["Lag_1h"] = df_temp.iloc[-4][TARGET_COL]
        if "Lag_24h" in FEATURES and len(df_temp) >= 96:
            row["Lag_24h"] = df_temp.iloc[-96][TARGET_COL]

        # 3. Wetter-Features von echten Daten (lag 15min)
        if USE_WEATHER and next_ts in df_full.index:
            for feat in WEATHER_FEATURES:
                row[feat] = df_full.loc[next_ts, feat]

        # 4. Vorhersage
        X = pd.DataFrame([row], index=[next_ts])[NUMERIC_FEATURES]
        y_hat = pipe.predict(X)[0]
        preds.append((next_ts, y_hat))

        # 5. Vorhersage zur Historie hinzufügen (für nächste Iteration)
        df_temp.loc[next_ts] = {TARGET_COL: y_hat}

    return pd.DataFrame(preds, columns=["ts", "forecast"]).set_index("ts")
```

### Warum ist das spannend?

1. **Selbst-referenzierend**: Jede Vorhersage wird zur Basis für die nächste
2. **Fehler-Akkumulation**: Kleine Fehler können sich über 96 Schritte verstärken
3. **Feature-Synchronisation**: Kalender (berechnet), Lags (vorherige Predictions), Wetter (echt)
4. **Sin/Cos-Berechnung on-the-fly** für zyklische Features

### Performance-Trick

```python
# multistep_forecast_recursive.py, Zeilen 52-56

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR
while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent
```

Automatisches Finden des Projekt-Roots → funktioniert unabhängig vom Ausführungsort!

---

## 🛠️ Feature Engineering

### Automatische Lag-Feature-Erstellung

```python
# data_preparation.py, Zeilen 183-189

# Lag Features (Stromverbrauch)
merged_15["Lag_15min"] = merged_15["Stromverbrauch"].shift(1)
merged_15["Lag_30min"] = merged_15["Stromverbrauch"].shift(2)
merged_15["Lag_1h"] = merged_15["Stromverbrauch"].shift(4)   # 4 × 15min = 1h
merged_15["Lag_24h"] = merged_15["Stromverbrauch"].shift(96)  # 96 × 15min = 24h
merged_15["Grundversorgte Kunden_Lag_15min"] = merged_15["Grundversorgte Kunden"].shift(1)
merged_15["Freie Kunden_Lag_15min"] = merged_15["Freie Kunden"].shift(1)
```

**Warum `shift(96)` für 24h?**
- 1h = 4 Intervalle (15min × 4)
- 24h = 96 Intervalle
- `shift(96)` holt den Wert von vor exakt 24h

### Wetter-Lag-Features

```python
# data_preparation.py, Zeilen 223-226

# Wetter-Lag = Werte 15min früher
for col in weather_cols:
    if col in merged_15.columns:
        merged_15[f"{col}_lag15"] = merged_15[col].shift(1)
```

**Kritisch!** Wetterdaten werden 15min verzögert → kein Data Leakage (siehe nächster Abschnitt)

### Differenz-Feature

```python
# data_preparation.py, Zeile 261

merged_15["Diff_15min"] = merged_15["Lag_15min"] - merged_15["Lag_30min"]
```

Erfasst die **Verbrauchsänderung** → wichtig für Trend-Erkennung!

---

## 🚫 Data Leakage Prevention

### Problem

Ein Prognosemodell darf **keine Informationen aus der Zukunft** verwenden!

**Beispiel (falsch)**:
```python
# FALSCH! Data Leakage
X["Temperatur"] = df.loc[timestamp, "Temperatur"]  # Temperatur zur gleichen Zeit
y = df.loc[timestamp, "Stromverbrauch"]
```

Zur Zeit `t` kennt man die Temperatur von `t` noch nicht!

### Lösung: Lag-Features

```python
# RICHTIG! Nur vergangene Daten
X["Temperatur_lag15"] = df.loc[timestamp - 15min, "Temperatur"]
X["Lag_15min"] = df.loc[timestamp - 15min, "Stromverbrauch"]
y = df.loc[timestamp, "Stromverbrauch"]
```

**Im Code**:
```python
# data_preparation.py, Zeilen 228-257

# Originale Meteodaten entfernen (nur Lags behalten)
original_weather_cols = [
    'Globalstrahlung; Zehnminutenmittel',
    'Diffusstrahlung; Zehnminutenmittel',
    # ... alle Wetter-Features
]

# Originale Wetterspalten entfernen
merged_15.drop(columns=[col for col in original_weather_cols if col in merged_15.columns],
               inplace=True)
```

Nur die `_lag15`-Varianten bleiben → **kein Data Leakage möglich**!

---

## 🔄 Sin/Cos-Encoding für zyklische Features

### Problem

Klassisches One-Hot-Encoding für zyklische Daten ist problematisch:

- **Monat 12 (Dezember)** und **Monat 1 (Januar)** sind sich ähnlich (Winter)
- One-Hot würde sie als komplett verschieden behandeln
- Numerisch (1, 2, ..., 12) impliziert falsche Ordnung (12 ≠ doppelt so viel wie 6)

### Lösung: Trigonometrische Kodierung

```python
# multistep_forecast_recursive.py, Zeilen 103-110

df["Monat_sin"] = np.sin(2 * np.pi * df["Monat"] / 12)
df["Monat_cos"] = np.cos(2 * np.pi * df["Monat"] / 12)
df["Wochentag_sin"] = np.sin(2 * np.pi * df["Wochentag"] / 7)
df["Wochentag_cos"] = np.cos(2 * np.pi * df["Wochentag"] / 7)
df["Stunde_sin"] = np.sin(2 * np.pi * df["Stunde (Lokal)"] / 24)
df["Stunde_cos"] = np.cos(2 * np.pi * df["Stunde (Lokal)"] / 24)
df["TagJahr_sin"] = np.sin(2 * np.pi * df["Tag des Jahres"] / 365)
df["TagJahr_cos"] = np.cos(2 * np.pi * df["Tag des Jahres"] / 365)
```

### Warum ist das brillant?

**Visualisierung (Einheitskreis)**:

```
        90° (Monat 3 = März)
             |
180° --------+-------- 0° (Monat 0/12)
(Monat 6)    |
            270° (Monat 9)
```

- **Dezember (12)** und **Januar (1)** liegen nahe beieinander auf dem Kreis!
- sin/cos-Werte sind kontinuierlich
- Modell kann Ähnlichkeit lernen

**Beispiel-Werte**:
| Monat | sin(Monat) | cos(Monat) |
|-------|-----------|-----------|
| 1 (Jan) | 0.50 | 0.87 |
| 12 (Dez) | -0.50 | 0.87 |
| 6 (Jun) | 0.00 | -1.00 |

→ Januar und Dezember haben **ähnliche cos-Werte** (beide ~0.87)!

### Helper-Funktion im rekursiven Forecast

```python
# multistep_forecast_recursive.py, Zeilen 48-49

def _sin_cos(value, period):
    return np.sin(2 * np.pi * value / period), np.cos(2 * np.pi * value / period)
```

Verwendet während der Vorhersage, um Kalender-Features on-the-fly zu berechnen!

---

## 🏗️ Pipeline-Architektur

### Scikit-Learn Pipeline

```python
# 1-step_forecast.py, Zeilen 47-51

preprocessor = ColumnTransformer(
    transformers=[("num", "passthrough", NUMERIC_FEATURES)],
    remainder="drop"
)
```

**Was macht das?**
- Wählt nur numerische Features aus
- Verwirft alle anderen (Text, kategorische ohne Encoding)
- `passthrough` = keine weitere Transformation (Daten bereits skaliert/encoded)

### Model Pipeline

```python
# 1-step_forecast.py, Zeilen 99-102

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", model),
])
```

**Vorteile**:
1. **Ein Objekt** für Preprocessing + Modell
2. **Persistierung** mit `joblib.dump(pipe, path)` speichert beides
3. **Reproduzierbarkeit**: Gleiche Transformationen bei Prediction

### Modell-Speicherung

```python
# 1-step_forecast.py, Zeilen 118-120

model_path = MODEL_DIR / f"{name}_1step_pipeline.joblib"
joblib.dump(pipe, model_path)
print("Gespeichert:", model_path)
```

Gespeichert wird **die gesamte Pipeline**, nicht nur das Modell!

Beim Laden:
```python
pipe = joblib.load("models/lgbm_1step_pipeline.joblib")
predictions = pipe.predict(X_new)  # Preprocessing automatisch angewendet!
```

---

## 🎯 Feature-Reduktion & Performance

### Ausschluss redundanter Features

```python
# multistep_forecast_recursive.py, Zeilen 81-101

EXCLUDE_WEATHER = [
    'Böenspitze (3-Sekundenböe); Maximum in km/h_lag15',  # Redundant zu m/s
    'Luftdruck reduziert auf Meeresniveau (QFF)_lag15',   # Redundant zu Barometerhöhe
    'Lufttemperatur 2 m ü. Gras_lag15',                   # Redundant zu 2m ü. Boden
    'Windgeschwindigkeit; Zehnminutenmittel in km/h_lag15', # Redundant zu m/s
    # ...
]

EXCLUDE_LAGS = [
    "Grundversorgte Kunden_Lag_15min",  # Schwache Prädiktoren
    "Freie Kunden_Lag_15min",
    "Lag_15min",  # Zu kurz für 24h-Forecast
    "Lag_30min",
    "Diff_15min"
]
```

**Warum?**
- **Multikollinearität** reduzieren (z.B. Wind in km/h vs. m/s)
- **Modell-Komplexität** senken
- **Overfitting** vermeiden
- **Performance** verbessern (weniger Features = schneller)

---

## 📦 Modular Design: Common Module

### Shared Code

```python
# common/data.py & common/metrics.py

import common.data as data
import common.metrics as m

load_dataset_and_split = data.load_dataset_and_split
train_test_metrics = m.train_test_metrics
```

**Vorteile**:
- **DRY-Prinzip** (Don't Repeat Yourself)
- Änderungen an Datenladung nur an einer Stelle
- Konsistente Metriken über alle Modelle

### Dynamisches Root-Finding

```python
# common/data.py (implizit)

ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and ROOT.name != "Yaren":
    ROOT = ROOT.parent
```

Findet automatisch das Yaren-Verzeichnis → funktioniert unabhängig von Ausführungsort!

---

## 🧪 Evaluation-Strategie

### 24h-Block-Evaluation

```python
# multistep_forecast_recursive.py, Zeilen 273-308

def eval_recursive_midnight_blocks(starts: pd.DatetimeIndex, label: str):
    maes, rmses, r2s, mapes, kept = [], [], [], [], []

    for start_ts in starts:
        # History: 7 Tage vor start_ts
        history_end = start_ts - pd.Timedelta("15min")
        history_start = history_end - pd.Timedelta(days=HISTORY_DAYS)
        history = df.loc[history_start:history_end].copy()

        # 24h-Forecast
        fc = forecast_multistep(pipe, history, df_full=df, steps=HORIZON)

        # Evaluation
        truth = df.reindex(fc.index)[TARGET_COL]
        if truth.isna().sum() > MAX_MISSING_PER_BLOCK:
            continue  # Skip unvollständige Tage

        mae, rmse, r2, mape = metrics_1d(truth.values, fc["forecast"].values)
        maes.append(mae)
        # ...

    return pd.DataFrame({"MAE": maes, "RMSE": rmses, ...})
```

**Warum ist das robust?**
1. **Realistische Bedingungen**: Forecast startet immer um 00:00 (lokal)
2. **Rolling Evaluation**: Nicht nur ein Tag, sondern alle Tage im Test-Set
3. **Missing-Data-Handling**: Tage mit >4 fehlenden Werten werden übersprungen
4. **Statistiken**: Mean/Median über alle Tage → robustere Metrik

---

## 🚀 Performance-Optimierungen

### LightGBM-Konfiguration

```python
# multistep_forecast_recursive.py, Zeilen 156-165

LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    force_col_wise=True  # ← Performance-Boost!
)
```

**force_col_wise=True**:
- Optimiert für spaltenweise Operationen
- Schneller bei vielen Features
- Reduziert Memory-Footprint

**n_jobs=-1**:
- Nutzt alle CPU-Kerne
- Parallelisierung über Bäume

---

## 🎓 Best Practices im Code

### 1. Type Hints (implizit durch docstrings)

```python
def metrics_1d(y_true: np.ndarray, y_pred: np.ndarray):
    """Berechnet Regressions-Metriken."""
    mae = mean_absolute_error(y_true, y_pred)
    return float(mae), float(rmse), float(r2), float(mape)
```

### 2. Defensive Programming

```python
# Prüfen, ob Features vorhanden sind
if col in merged_15.columns:
    merged_15[f"{col}_lag15"] = merged_15[col].shift(1)
```

### 3. Magic Numbers vermeiden

```python
HORIZON = 96           # 24h in 15min-Intervallen
HISTORY_DAYS = 7       # 7 Tage Historie für Forecast
MAX_MISSING_PER_BLOCK = 4  # Max. fehlende Werte pro Tag
```

Besser als hartcodierte `96`, `7`, `4` im Code!

---

## 🔍 Debugging-Tipps

### Zeitstempel-Debugging

```python
print("Index tz:", merged_15.index.tz)             # Zeitzone prüfen
print("Duplikate:", merged_15.index.duplicated().sum())  # Doppelte Timestamps
full_idx = pd.date_range(merged_15.index.min(), merged_15.index.max(), freq="15min", tz="UTC")
print("Fehlende Slots:", len(full_idx.difference(merged_15.index)))  # Lücken
```

### Feature-Debugging

```python
print(merged_15[[
    "IstArbeitstag", "IstSonntag",
    "Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h", "Diff_15min"
]].head())
```

---

## 📚 Weiterführende Ressourcen

- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Feature Engineering for Time Series](https://www.kaggle.com/learn/time-series)
- [Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)

---

**← Zurück zum [README](../README.md) | Weiter zu [Results.md](Results.md) →**
