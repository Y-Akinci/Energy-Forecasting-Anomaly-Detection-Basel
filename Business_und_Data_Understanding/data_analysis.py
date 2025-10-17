import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from utils import helpers

df = helpers.csv_file()

print(df.index.dtype)
#print(df.head(3))

# Erwartetes Zeitraster (alle 15 Minuten)
start = df.index.min()
ende = df.index.max()

# Alle Zeitpunkte, die theoretisch vorhanden sein sollten
voller_index = pd.date_range(start=start, end=ende, freq="15T", tz="UTC")

# Fehlende Zeitpunkte
fehlende = voller_index.difference(df.index)

# Doppelte Zeitpunkte
doppelte = df.index[df.index.duplicated()]

print("Zeitraum:", start, "bis", ende)
print("Sollte haben:", len(voller_index), "Zeitpunkte")
print("Hat tatsächlich:", len(df.index))
print("Fehlende Zeitpunkte:", len(fehlende))
print("Doppelte Zeitpunkte:", len(doppelte))

# Fehlende Zeitpunkte anzeigen
print("Erste 10 fehlende Zeitpunkte:")
print(fehlende)

# Optional: Differenzen zwischen Zeitpunkten prüfen.
delta = df.index.to_series().diff().value_counts().head(5)
print("\nHäufigste Zeitabstände:")
print(delta)





