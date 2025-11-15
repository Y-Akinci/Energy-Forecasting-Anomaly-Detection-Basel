import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import pandas as pd

# Pfade
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
F1 = os.path.join(DATA_DIR, "raw data", "251107_Wetterdaten_Basel_2010-2019.csv")
F2 = os.path.join(DATA_DIR, "raw data", "251107_Wetterdaten_Basel_2020-2024.csv")


OUT_MERGED_10 = os.path.join(DATA_DIR, "wetter_merged_10min.csv")
OUT_MERGED_15 = os.path.join(DATA_DIR, "wetter_merged_15min.csv")

# ---------------------------
# 1) Wetter laden (beide CSV)
# ---------------------------
# Meteo-Schweiz ist meist latin1 + Semikolon
w1 = pd.read_csv(F1, sep=";", decimal=",", encoding="latin1", low_memory=False)
w2 = pd.read_csv(F2, sep=";", decimal=",", encoding="latin1", low_memory=False)
for df in (w1, w2):
    df.columns = [c.strip() for c in df.columns]

# Zeitstempel bilden -> **direkt in UTC** parsen (kein DST-Drama)
def make_ts_utc(df):
    if "Date and Time" in df.columns:
        return pd.to_datetime(df["Date and Time"], errors="coerce", dayfirst=True, utc=True)
    elif {"Date", "Time"}.issubset(df.columns):
        return pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce", dayfirst=True, utc=True
        )
    else:
        raise ValueError("Zeitspalte fehlt: erwarte 'Date and Time' oder 'Date'+'Time'.")

w1["DateTime"] = make_ts_utc(w1)
w2["DateTime"] = make_ts_utc(w2)

w1 = w1.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
w2 = w2.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()

# Mergen und ueberlappende Timestamps bereinigen (letzten behalten)
weather = pd.concat([w1, w2], axis=0)
weather = weather[~weather.index.duplicated(keep="last")].sort_index()

print("MERGED Wetter (roh):", weather.shape)
print("Zeitraum:", weather.index.min(), "bis", weather.index.max())

# ------------------------------------------------
# 2) Analyse der **rohen** Wetterdaten (10-Minuten)
# ------------------------------------------------
# Haeufige Zeitabstaende
delta_counts = weather.index.to_series().diff().value_counts().head(5)
print("\nHaeufigste Zeitabstaende (roh):")
print(delta_counts)

# Erwartetes 10-Min-Raster (UTC)
start_utc = weather.index.min()
end_utc = weather.index.max()
full_10 = pd.date_range(start=start_utc, end=end_utc, freq="10T", tz="UTC")

# Fehlende und doppelte Zeitpunkte
missing_10 = full_10.difference(weather.index)
dups_10 = weather.index[weather.index.duplicated()]

print("\nSollte haben (10T):", len(full_10), "Zeitpunkte")
print("Hat tatsaechlich:", len(weather.index))
print("Fehlende (10T):", len(missing_10))
print("Doppelte (10T):", len(dups_10))

print("\nErste 20 fehlende (10T):")
print(missing_10[:20])

# Spalten-Qualitaet
nonnull_pct = weather.notna().mean().sort_values(ascending=True) * 100
print("\nNicht-NA-Quote pro Spalte (%) – niedrigste zuerst:")
print(nonnull_pct.head(15).round(2))

# Speichern der **rohen** gemergten Wetterdaten (10-Min)
weather.to_csv(OUT_MERGED_10, sep=";", encoding="utf-8")
print("\nOK ->", OUT_MERGED_10)

# --------------------------------------------------------------------
# 3) Auf **15-Minuten** mappen (fuer spaetere Joins mit Strom sinnvoll)
#    - Numerisch: zeitinterpoliert
#    - Nicht-numerisch: nearest mit Toleranz 7:30 min, Rander gefuellt
# --------------------------------------------------------------------
full_15 = pd.date_range(start=start_utc, end=end_utc, freq="15T", tz="UTC")

num_cols = weather.select_dtypes(include="number").columns
other_cols = weather.columns.difference(num_cols)

num_15 = (
    weather[num_cols]
    .reindex(full_15)                                # exakt 15-Min Achse
    .interpolate(method="time", limit_direction="both")
)

if len(other_cols) > 0:
    target = pd.DataFrame(index=full_15).sort_index()
    other_15 = pd.merge_asof(
        target,
        weather[other_cols].sort_index(),
        left_index=True, right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("7min30s")            # 10->15 min: halbe Distanz
    ).ffill().bfill()
    weather_15 = pd.concat([num_15, other_15], axis=1)
else:
    weather_15 = num_15

# Analyse 15-Min
delta_15 = weather_15.index.to_series().diff().value_counts().head(5)
missing_15 = full_15.difference(weather_15.index)
dups_15 = weather_15.index[weather_15.index.duplicated()]

print("\n[15T] Haeufigste Abstaende:")
print(delta_15)
print("[15T] Erwartet:", len(full_15), " | Hat:", len(weather_15.index))
print("[15T] Fehlende:", len(missing_15), " | Doppelte:", len(dups_15))

nonnull_pct_15 = weather_15.notna().mean().sort_values(ascending=True) * 100
print("\n[15T] Nicht-NA-Quote pro Spalte (%) – niedrigste zuerst:")
print(nonnull_pct_15.head(15).round(2))

# Speichern der 15-Min-Version
weather_15.to_csv(OUT_MERGED_15, sep=";", encoding="utf-8")
print("\nOK ->", OUT_MERGED_15)

