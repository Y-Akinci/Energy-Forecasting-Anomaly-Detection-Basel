import pandas as pd
import os, sys
def csv_file():
    dateipfad = r"C:\Users\User\Documents\GitHub\Forecast & Anomalie Detection\251006_Stromverbrauch Basel 2012-2025.csv"

    df = pd.read_csv(dateipfad, sep = ";")

    print("Anzahl Zeilen und Spalten:", df.shape)
    print("Spaltennamen:", df.columns.tolist())
    #print(df.head())

    df["Start der Messung"] = pd.to_datetime(df["Start der Messung"], errors = "coerce")
    df = df.set_index("Start der Messung")
    return df