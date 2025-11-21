import pandas as pd
import numpy as np

# Anzeigeoptionen  für bessere Übersicht (nicht für alle)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)

# Datei einlesen 
file_path = "merged_strom_meteo_30min_sync.csv"  # Dateiname muss geändert werden, diese version ist mit haris seiner version funktionfähig
data = pd.read_csv(file_path, sep=";", encoding="utf-8")

# Spalten löschen
columns_to_drop = [
    "DateTime",
    "station_abbr",
    "Start der Messung",
    "Start der Messung (Text)",
    "Date and Time"
]
# Sicher löschen nur wenn Spalte existiert
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# zeilen löschen bis und mit 75983
data = data.drop(index=range(0, 75983))

data.to_excel("merged_strom_meteo_30min_sync_clean.xlsx", index=False)