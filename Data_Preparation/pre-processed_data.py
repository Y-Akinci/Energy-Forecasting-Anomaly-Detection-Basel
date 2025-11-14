import pandas as pd
import os

# --- Datei laden ---
DATA_PATH = r"c:\Users\User\VisualStudioCode\EnergyForecastAnomalyDetection\data\merged_strom_meteo_15min.csv"

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8", parse_dates=[0], index_col=0)

print("Datei geladen:", df.shape, "Zeilen x Spalten")

# --- Fehlende Werte zählen ---
nan_count = df.isna().sum()
nan_pct = (df.isna().mean() * 100).round(2)

report = pd.DataFrame({
    "NaN_Anzahl": nan_count,
    "NaN_%": nan_pct
}).sort_values("NaN_%", ascending=False)

print("\n--- Fehlende Werte pro Spalte ---")
print(report.head(20))

# --- Gesamtergebnis ---
total_missing = df.isna().sum().sum()
print(f"\nGesamt fehlende Zellen: {total_missing:,}")
