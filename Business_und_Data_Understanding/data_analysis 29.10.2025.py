from pydoc import Helper
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from utils import helpers


file_path = "251006_StromverbrauchBasel2012-2025.csv"

# Excel einlesen (automatisch erstes Sheet)
df = helpers.csv_file()

print("=== Vorschau auf die Daten ===")
print(df.head())
print("\nForm:", df.shape)
print("Spalten:", list(df.columns))

# ======================
# 2) BASIS-ÜBERSICHT
# ======================
print("\n=== Allgemeine Informationen ===")
df.info()

print("\n=== Fehlende Werte (pro Spalte) ===")
print(df.isnull().sum())

print("\n=== Beschreibende Statistik (numerische Spalten) ===")
print(df.describe().T)  # T = transponiert für bessere Lesbarkeit

# ======================
# 3) ERWEITERTE ANALYSE
# ======================
print("\n=== Höchstwerte und Tiefstwerte (pro Spalte) ===")
for col in df.select_dtypes(include=['number']).columns:
    print(f"{col}: min={df[col].min()} | max={df[col].max()} | mean={df[col].mean():.2f}")

# ======================
# 4) KORRELATIONEN (nur numerische Spalten)
# ======================
print("\n=== Korrelationsmatrix ===")
corr = df.corr(numeric_only=True)
print(corr)

# ======================
# 5) OPTIONAL: TOP-WERTE ZEIGEN
# ======================
print("\n=== Zeilen mit Höchstwert in jeder Spalte ===")
for col in df.select_dtypes(include=['number']).columns:
    idx = df[col].idxmax()
    print(f"{col}: Index={idx}, Wert={df.loc[idx, col]}")

# ======================
# 6) SPEICHERN DER ANALYSE
# ======================
summary = {
    "Anzahl Zeilen": len(df),
    "Anzahl Spalten": len(df.columns),
    "Fehlende Werte (gesamt)": df.isnull().sum().sum(),
    "Numerische Spalten": list(df.select_dtypes(include=['number']).columns)
}

pd.DataFrame(summary.items(), columns=["Beschreibung", "Wert"]).to_excel("data_understanding_summary.xlsx", index=False)
print("\nAnalyse abgeschlossen ✅ — Datei 'data_understanding_summary.xlsx' gespeichert.")






