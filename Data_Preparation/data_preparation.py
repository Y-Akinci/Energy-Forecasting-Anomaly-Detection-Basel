import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import helpers
import pandas as pd

# Pfade
TZ = "Europe/Zurich"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "Energy-Forecasting-Anomaly-Detection-Basel", "data")
OUT_15 = os.path.join(DATA_DIR, "merged_strom_meteo_15min.csv")

# === 1) STROM laden (helpers -> Europe/Zurich) und auf UTC bringen ===
power_local = helpers.csv_file()      # tz-aware Europe/Zurich
power = power_local.tz_convert("UTC")      # alles in UTC gegen DST-Probleme

# Exakte 15-min Achse auf Basis der Stromdaten (LEFT-Master-Achse)
full_idx_utc = pd.date_range(power.index.min(), power.index.max(), freq="15min", tz="UTC")
power_15 = power.reindex(full_idx_utc)     # LEFT: evtl. NaNs bleiben NaNs

# === 2) WETTER laden (2 CSVs), direkt in UTC parsen ===
f1 = os.path.join(DATA_DIR, "raw data", "251107_Wetterdaten_Basel_2010-2019.csv")
f2 = os.path.join(DATA_DIR, "raw data", "251107_Wetterdaten_Basel_2020-2024.csv")

w1 = pd.read_csv(f1, sep=";", decimal=",", encoding="latin1", low_memory=False)
w2 = pd.read_csv(f2, sep=";", decimal=",", encoding="latin1", low_memory=False)
for df in (w1, w2):
    df.columns = [c.strip() for c in df.columns]

def make_ts_utc(df):
    if "Date and Time" in df.columns:
        ts = pd.to_datetime(df["Date and Time"], errors="coerce", dayfirst=True)
    else:
        raise ValueError("Zeitspalte fehlt: erwarte 'Date and Time'")

    ts = pd.to_datetime(df["Date and Time"], dayfirst=True, utc=True)

    return ts

w1["DateTime"] = make_ts_utc(w1)
w2["DateTime"] = make_ts_utc(w2)

w1 = w1.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
w2 = w2.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()

# === DATA COLLECTON ===
# Mergen + doppelte Timestamps entfernen
weather = pd.concat([w1, w2], axis=0)
weather = weather[~weather.index.duplicated(keep="last")].sort_index()

# === 3) 10-min -> 15-min ===
# Numerische Spalten: zeitbasierte Interpolation exakt auf die 15-min Strom-Achse
num_cols = weather.select_dtypes(include="number").columns
num_15 = weather[num_cols].reindex(full_idx_utc).interpolate(method="time", limit_direction="both")


# Nicht-numerische Spalten: nächster Wert (nearest) ohne Verschiebung
other_cols = weather.columns.difference(num_cols)
if len(other_cols) > 0:
    target = pd.DataFrame(index=full_idx_utc).sort_index()
    # merge_asof mit Toleranz 7:30 min (zwischen 10-min Stützstellen)
    other_15 = pd.merge_asof(
        target, 
        weather[other_cols].sort_index(),
        left_index=True, right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("7min30s")
    )
    # Ränder auffüllen
    other_15 = other_15.ffill().bfill()
    weather_15 = pd.concat([num_15, other_15], axis=1)
else:
    weather_15 = num_15

# === 4) LEFT-Join auf Strom (keine zeitliche Verschiebung) ===
merged_15 = power_15.join(weather_15, how="left")

print("Index tz:", merged_15.index.tz)             # sollte UTC zeigen
print("Duplikate:", merged_15.index.duplicated().sum())  # 0
full_idx = pd.date_range(merged_15.index.min(), merged_15.index.max(), freq="15min", tz="UTC")
print("Fehlende Slots:", len(full_idx.difference(merged_15.index)))  # 0 oder nur echte Datenlücken

# === DATA PREPROCCESSING ===

# === DATA CLEANING ===

# Zeilen löschen, welche keinen Wert bei der Zielvariable "Stromverbrauch" haben
merged_15.dropna(subset="Stromverbrauch", inplace=True)

# === DATATYPES ===

# Datentyp ändern, integers
int_cols = [
    "Jahr", "Monat", "Tag", "Wochentag", "Tag des Jahres",
    "Quartal", "Woche des Jahres",
]

for col in int_cols:
    merged_15[col] = pd.to_numeric(
        merged_15[col], errors="coerce"
    ).astype("Int64") 

# Datentypen ändern, float64
float_cols = [
    'Stromverbrauch',
    'Grundversorgte Kunden',
    'Freie Kunden',
    'Globalstrahlung; Zehnminutenmittel',
    'Diffusstrahlung; Zehnminutenmittel',
    'Langwellige Einstrahlung; Zehnminutenmittel',
    'Sonnenscheindauer; Zehnminutensumme',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in km/h',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in m/s',
    'BÃ¶enspitze (SekundenbÃ¶e); Maximum in km/h',
    'Böenspitze (Sekundenböe)',
    'Chilltemperatur',
    'Dampfdruck 2 m ü. Boden',
    'Luftdruck reduziert auf Meeresniveau (QFF)',
    'Luftdruck reduziert auf Meeresniveau mit Standardatmosphäre (QNH)',
    'Luftrdruck auf Barometerhöhe',
    'Lufttemperatur 2 m ü. Boden',
    'Lufttemperatur 2 m ü. Gras',
    'Lufttemperatur Bodenoberfläche',
    'Niederschlag; Zehnminutensumme',
    'Taupunkt 2 m ü. Boden',
    'Windgeschwindigkeit skalar; Zehnminutenmittel in m/s',
    'Windgeschwindigkeit vektoriell',
    'Windgeschwindigkeit; Zehnminutenmittel in km/h',
    'Windrichtung; Zehnminutenmittel',
    'relative Luftfeuchtigkeit'
]

for col in float_cols:
    merged_15[col] = pd.to_numeric(merged_15[col], errors="coerce").astype("float64")

# Datentypen für Kategoriale variablen
categorical_features = [
    "Jahr",
    "Monat",
    "Wochentag",
    "Quartal",
    "Woche des Jahres"
]

for col in categorical_features:
    merged_15[col] = merged_15[col].astype("category")

# FEATURE ENGINEERING
## Feature CREATION

# Lokale Zeit aus UTC-Index erzeugen
idx_local = merged_15.index.tz_convert("Europe/Zurich")
# Lokale Datum- und Zeitspalten erstellen
merged_15["Datum (Lokal)"] = idx_local.date
merged_15["Zeit (Lokal)"] = idx_local.time
merged_15["Stunde (Lokal)"] = idx_local.hour
merged_15["Stunde (Lokal)"] = pd.to_numeric(merged_15["Stunde (Lokal)"], errors="coerce").astype("int64")
# Index-Spalte erzeugen, damit rename funktionieren kann
merged_15.reset_index(inplace=True)
# Index-Spalte korrekt umbenennen
merged_15.rename(columns={"index": "Start der Messung (UTC)"}, inplace=True)
# Index wieder setzen
merged_15.set_index("Start der Messung (UTC)", inplace=True)
# Datentyp der lokalen Spalten ändern
# --- 1) Jahreszeiten ---
#def jahreszeit(month):
    #if month in [12, 1, 2]:
        #return "1"
    #elif month in [3, 4, 5]:
        #return "2"
    #elif month in [6, 7, 8]:
        #return "3"
    #else:
        #return "4"

#merged_15["Jahreszeit"] = merged_15["Monat"].map(jahreszeit).astype("category"))
wd = merged_15["Wochentag"].astype("int")

# Arbeitstag vs Sonntag
merged_15["IstSonntag"] = (wd == 6).astype(int)
merged_15["IstArbeitstag"] = (wd < 5).astype(int)

# Lag Features (Stromverbrauch)
merged_15["Lag_15min"] = merged_15["Stromverbrauch"].shift(1)
merged_15["Lag_30min"] = merged_15["Stromverbrauch"].shift(2)
merged_15["Lag_1h"] = merged_15["Stromverbrauch"].shift(4)
merged_15["Lag_24h"] = merged_15["Stromverbrauch"].shift(96)

# Zeilen löschen, die durch lag NaN Werte haben
lag_cols = ["Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h"]
merged_15.dropna(subset=lag_cols, inplace=True)

# --- 4) Wetter-Lag Features (nur Meteodaten laggen!) ---

# Liste der Wetterspalten (aus deinen float_cols)
weather_cols = [
    'Globalstrahlung; Zehnminutenmittel',
    'Diffusstrahlung; Zehnminutenmittel',
    'Langwellige Einstrahlung; Zehnminutenmittel',
    'Sonnenscheindauer; Zehnminutensumme',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in km/h',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in m/s',
    'BÃ¶enspitze (SekundenbÃ¶e); Maximum in km/h',
    'Böenspitze (Sekundenböe)',
    'Chilltemperatur',
    'Dampfdruck 2 m ü. Boden',
    'Luftdruck reduziert auf Meeresniveau (QFF)',
    'Luftdruck reduziert auf Meeresniveau mit Standardatmosphäre (QNH)',
    'Luftrdruck auf Barometerhöhe',
    'Lufttemperatur 2 m ü. Boden',
    'Lufttemperatur 2 m ü. Gras',
    'Lufttemperatur Bodenoberfläche',
    'Niederschlag; Zehnminutensumme',
    'Taupunkt 2 m ü. Boden',
    'Windgeschwindigkeit skalar; Zehnminutenmittel in m/s',
    'Windgeschwindigkeit vektoriell',
    'Windgeschwindigkeit; Zehnminutenmittel in km/h',
    'Windrichtung; Zehnminutenmittel',
    'relative Luftfeuchtigkeit'
]

# Wetter-Lag = Werte 15min früher
for col in weather_cols:
    if col in merged_15.columns:  # falls Spalte existiert
        merged_15[f"{col}_lag15"] = merged_15[col].shift(1)

# Originale Meteodaten entfernen (nur Lags behalten)

# Liste der originalen Wetterspalten (deine float_cols)
original_weather_cols = [
    'Globalstrahlung; Zehnminutenmittel',
    'Diffusstrahlung; Zehnminutenmittel',
    'Langwellige Einstrahlung; Zehnminutenmittel',
    'Sonnenscheindauer; Zehnminutensumme',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in km/h',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in m/s',
    'BÃ¶enspitze (SekundenbÃ¶e); Maximum in km/h',
    'Böenspitze (Sekundenböe)',
    'Chilltemperatur',
    'Dampfdruck 2 m ü. Boden',
    'Luftdruck reduziert auf Meeresniveau (QFF)',
    'Luftdruck reduziert auf Meeresniveau mit Standardatmosphäre (QNH)',
    'Luftrdruck auf Barometerhöhe',
    'Lufttemperatur 2 m ü. Boden',
    'Lufttemperatur 2 m ü. Gras',
    'Lufttemperatur Bodenoberfläche',
    'Niederschlag; Zehnminutensumme',
    'Taupunkt 2 m ü. Boden',
    'Windgeschwindigkeit skalar; Zehnminutenmittel in m/s',
    'Windgeschwindigkeit vektoriell',
    'Windgeschwindigkeit; Zehnminutenmittel in km/h',
    'Windrichtung; Zehnminutenmittel',
    'relative Luftfeuchtigkeit'
]

# Originale Wetterspalten entfernen
merged_15.drop(columns=[col for col in original_weather_cols if col in merged_15.columns],
               inplace=True)

# Originale Meteodaten entfernt – nur Lag-Wetterdaten bleiben im Dataset!")

# Differenz zum letzten Verbrauch
merged_15["Diff_15min"] = merged_15["Lag_15min"] - merged_15["Lag_30min"]

print("Feature Engineering erfolgreich abgeschlossen!")
print(merged_15[[
    "IstArbeitstag", "IstSonntag",
    "Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h", "Diff_15min"
]].head())

# FEATURE SELECTION
# Tag, Monat, und Jahr entfernt
delete_columns = ["Jahr", "Date", "Time", "Date and Time", "Start der Messung (Text)", "station_abbr"]
cols_to_drop = [c for c in delete_columns if c in merged_15.columns]
merged_15.drop(columns=cols_to_drop, inplace=True)

# EXPORT FINAL DATAFRAME

# Zielpfad setzen
output_path = os.path.join(DATA_DIR, "processed_merged_features.csv")

# CSV exportieren
merged_15.to_csv(output_path, sep=";", encoding="latin1")

print("Export erfolgreich!")
print("Datei gespeichert unter:", output_path)
print("Shape:", merged_15.shape)
