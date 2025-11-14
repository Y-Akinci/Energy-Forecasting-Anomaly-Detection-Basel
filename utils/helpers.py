import pandas as pd
import os

def csv_file(filename="251006_StromverbrauchBasel2012-2025.csv"):
    # Projekt-Root finden (eine Ebene über /utils)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, "data", "raw data", filename)


    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {data_path}")

    df = pd.read_csv(data_path, sep=";", low_memory=False)
    print("Anzahl Zeilen und Spalten:", df.shape)
    print("Spaltennamen:", df.columns.tolist())

    # Zeitstempel als Datetime setzen
    if "Start der Messung" in df.columns:
        df["Start der Messung"] = pd.to_datetime(df["Start der Messung"], errors="coerce")
        df = df.set_index("Start der Messung").sort_index()

    return df

