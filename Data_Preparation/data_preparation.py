import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import helpers
import pandas as pd
from pathlib import Path
import seaborn as sns

# Pfade
TZ = "Europe/Zurich"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "Energy-Forecasting-Anomaly-Detection-Basel", "data")
OUT_15 = os.path.join(DATA_DIR, "merged_strom_meteo_15min.csv")

# === 1) STROM laden (helpers -> Europe/Zurich) und auf UTC bringen ===
power_local = helpers.csv_file()           # tz-aware Europe/Zurich
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
    elif {"Date", "Time"}.issubset(df.columns):
        ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                            errors="coerce", dayfirst=True)
    else:
        raise ValueError("Zeitspalte fehlt: erwarte 'Date and Time' oder 'Date'+'Time'.")

    # 1) Strings sind lokale Zeiten -> als Europe/Zurich lokalisieren (keine Verschiebung),
    # 2) dann sauber nach UTC konvertieren (Verschiebung passiert hier)
    ts = ts.dt.tz_localize(
        "Europe/Zurich",
        nonexistent="shift_forward",  # Frühling: übersprungene Stunde
        ambiguous="NaT"               # Herbst: doppelte Stunde -> als NaT markieren (oder "infer")
    ).dt.tz_convert("UTC")
    return ts

w1["DateTime"] = make_ts_utc(w1)
w2["DateTime"] = make_ts_utc(w2)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

w1 = w1.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
w2 = w2.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()

# === DATA COLLECTON ===
# Mergen + doppelte Timestamps entfernen
weather = pd.concat([w1, w2], axis=0)
weather = weather[~weather.index.duplicated(keep="last")].sort_index()

# === 3) 10-min -> 15-min ===
# Numerische Spalten: zeitbasierte Interpolation exakt auf die 15-min Strom-Achse
num_cols = weather.select_dtypes(include="number").columns
num_15 = (
    weather[num_cols]
    .reindex(full_idx_utc)                       # exakt gleiche Achse wie Strom
    .interpolate(method="time", limit_direction="both")
)

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

# === 5) Export ===
#merged_15.to_csv(OUT_15, sep=";", encoding="utf-8")
#print("OK ->", OUT_15)
#print("Rows:", len(merged_15), "Cols:", merged_15.shape[1])

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

# Datentypen für KAtegoriale variablen
categorical_features = [
    "Jahr",
    "Monat",
    "Tag",
    "Wochentag",
    "Quartal",
    "Woche des Jahres"
]

for col in categorical_features:
    merged_15[col] = merged_15[col].astype("category")

# === FEATURE SELECTION ===

# Tag, Monat, und Jahr entfernt
merged_15.drop(["Jahr", "Monat","Tag"], axis=1)

# === FEATURE ENGINEERING ===

# "Date" und "Time" als eigene Spalten erstellen
merged_15.reset_index(inplace=True)
merged_15["Date"] = merged_15["index"].dt.strftime("%Y-%m-%d")
merged_15["Time"] = merged_15["index"].dt.strftime("%H-%M-%S")
# Spalte mit dem Tag im Jahr kreieren
merged_15['DayOfYear'] = merged_15["index"].dt.dayofyear
merged_15.set_index("index", inplace=True)

# Datentyp ändern, Zeitwerte
merged_15["Date"] = pd.to_datetime(merged_15["Date"], format="%Y-%m-%d")
merged_15["Time"] = pd.to_datetime(merged_15["Time"], format="%H-%M-%S")

# Spalte mit dem Tag im Jahr kreieren

print(merged_15.head())