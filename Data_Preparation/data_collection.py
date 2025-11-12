import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import pandas as pd
from utils import helpers

# Pfade
TZ = "Europe/Zurich"
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(ROOT), "data")
OUT_15 = os.path.join(DATA_DIR, "merged_strom_meteo_15min.csv")

# === 1) STROM laden (helpers -> Europe/Zurich) und auf UTC bringen ===
power_local = helpers.csv_file()           # tz-aware Europe/Zurich
power = power_local.tz_convert("UTC")      # alles in UTC gegen DST-Probleme

# Exakte 15-min Achse auf Basis der Stromdaten (LEFT-Master-Achse)
full_idx_utc = pd.date_range(power.index.min(), power.index.max(), freq="15min", tz="UTC")
power_15 = power.reindex(full_idx_utc)     # LEFT: evtl. NaNs bleiben NaNs

# === 2) WETTER laden (2 CSVs), direkt in UTC parsen ===
f1 = os.path.join(DATA_DIR, "251107_Wetterdaten_Basel_2010-2019.csv")
f2 = os.path.join(DATA_DIR, "251107_Wetterdaten_Basel_2020-2024.csv")

w1 = pd.read_csv(f1, sep=";", decimal=",", encoding="latin1", low_memory=False)
w2 = pd.read_csv(f2, sep=";", decimal=",", encoding="latin1", low_memory=False)
for df in (w1, w2):
    df.columns = [c.strip() for c in df.columns]

def make_ts_utc(df):
    if "Date and Time" in df.columns:
        return pd.to_datetime(df["Date and Time"], errors="coerce", dayfirst=True, utc=True)
    elif {"Date", "Time"}.issubset(df.columns):
        return pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                              errors="coerce", dayfirst=True, utc=True)
    else:
        raise ValueError("Zeitspalte fehlt: erwarte 'Date and Time' oder 'Date'+'Time'.")

w1["DateTime"] = make_ts_utc(w1)
w2["DateTime"] = make_ts_utc(w2)

w1 = w1.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
w2 = w2.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()

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

# Optional: Index zurück in Lokalzeit (nur Anzeige)
merged_15.index = merged_15.index.tz_convert(TZ)

# === 5) Export ===
merged_15.to_csv(OUT_15, sep=";", encoding="utf-8")
print("OK ->", OUT_15)
print("Rows:", len(merged_15), "Cols:", merged_15.shape[1])
