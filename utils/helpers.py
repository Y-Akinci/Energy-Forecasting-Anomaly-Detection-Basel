import pandas as pd
import os

def csv_file():
    # Projekt-Root finden (eine Ebene über /utils)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, "data", "stromverbrauch_basel.csv")

    df = pd.read_csv(data_path, sep=";")

    print("Anzahl Zeilen und Spalten:", df.shape)
    print("Spaltennamen:", df.columns.tolist())

    df["Start der Messung"] = pd.to_datetime(df["Start der Messung"], errors="coerce")
    df = df.set_index("Start der Messung")
    return df