"""
MODELING FRAMEWORK V11_PREDICT - ZUKUNFTS-PROGNOSE
==================================================
Lädt das beste trainierte Modell aus V11 (mit zyklischer Kodierung)
und generiert einen Forecast für einen definierten Zeitraum in der Zukunft.

WICHTIGER HINWEIS:
1. Zukünftige Wetterdaten: Dieses Skript verwendet die LETZTEN bekannten 
   Wetterwerte (Persistence) aus den historischen Daten als Annahme für die Zukunft. 
   Für eine echte, genaue Prognose MÜSSEN diese Werte durch einen echten 
   Wetter-Forecast ersetzt werden.
2. Lag_24h: Wird aus den letzten bekannten Stromverbrauchsdaten berechnet.
"""

import os
import pandas as pd
import numpy as np
from joblib import load
from datetime import timedelta

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # ACHTUNG: OUTPUT_DIR muss auf das Verzeichnis zeigen, 
    # in dem V11 das Modell gespeichert hat.
    MODEL_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Modeling\Haris\V11_Cyclic"
    DATA_DIR = r"C:\Users\haris\OneDrive\Anlagen\Desktop\ML nach Model\Energy-Forecasting-Anomaly-Detection-Basel\Data"
    
    # Historische Datei (wird nur geladen, um den Startpunkt und die letzten Lags zu bestimmen)
    INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv") 
    
    FORECAST_DURATION_DAYS = 7     # Wie viele Tage in die Zukunft soll prognostiziert werden
    TIME_STEP_MINUTES = 15         # Intervall der Daten (Muss 15 Minuten sein)
    
    # Ausgabepfad für die Prognose
    PREDICTION_OUTPUT_FILE = os.path.join(MODEL_DIR, "forecast_V11_7d.csv") 
    
    TARGET_COL = "Stromverbrauch"

# ============================================================
# HELPER FUNKTIONEN (Müssen identisch zu V11 sein!)
# ============================================================

def apply_cyclic_encoding(df, col_name, period):
    """
    Wendelt Sinus- und Cosinus-Transformation auf eine zyklische Spalte an.
    """
    if col_name in df.columns:
        data = df[col_name]
        # Sicherstellen, dass der Zähler bei 0 beginnt (0 bis period-1)
        if data.min() > 0 and data.max() == period:
            data = data - 1
        
        # Sinus und Cosinus Transformation
        df[col_name + '_Sin'] = np.sin(2 * np.pi * data / period)
        df[col_name + '_Cos'] = np.cos(2 * np.pi * data / period)
        
        # Entferne die ursprüngliche numerische Spalte
        df.drop(columns=[col_name], inplace=True)
    return df

def load_historical_data_and_model():
    """Lädt historische Daten, das Modell und die verwendeten Features."""
    # 1. Historische Daten laden (für Lags)
    try:
        df_hist = pd.read_csv(
            Config.INPUT_FILE,
            sep=";",
            encoding="latin1",
            index_col=0,
            parse_dates=True
        ).sort_index()
        print(f"Historische Daten geladen bis: {df_hist.index.max()}")
        
        # Nur relevante Spalten behalten (Target und Features für Lags)
        required_cols = [Config.TARGET_COL] 
        # Optional: Hier müssten die Spalten für Lag_24h und die Wetter-Lags
        # enthalten sein, um die letzten Werte zu extrahieren. Wir laden alle.

    except FileNotFoundError:
        print(f"FEHLER: Historische Datei nicht gefunden: {Config.INPUT_FILE}")
        exit()

    # 2. Modell und Features laden
    try:
        model_path = os.path.join(Config.MODEL_DIR, 'best_model.joblib')
        feature_path = os.path.join(Config.MODEL_DIR, 'features.npy')
        
        model = load(model_path)
        features = np.load(feature_path, allow_pickle=True).tolist()
        
        print(f"Modell ('{type(model).__name__}') und {len(features)} Features geladen.")
        
    except FileNotFoundError:
        print(f"FEHLER: Modell- oder Feature-Datei nicht gefunden. ")
        print(f"Bitte stellen Sie sicher, dass {Config.MODEL_DIR} existiert und die Dateien enthält.")
        exit()
        
    return df_hist, model, features

def prepare_future_data(df_hist, features):
    """
    Erstellt den DataFrame für die Prognose und füllt die Features.
    """
    last_hist_date = df_hist.index.max()
    
    # 1. Zukunfts-Index erstellen
    end_date = last_hist_date + timedelta(days=Config.FORECAST_DURATION_DAYS)
    future_index = pd.date_range(
        start=last_hist_date + timedelta(minutes=Config.TIME_STEP_MINUTES),
        end=end_date,
        freq=f'{Config.TIME_STEP_MINUTES}min'
    )
    
    df_future = pd.DataFrame(index=future_index)
    
    print(f"\nErzeuge Index von {future_index.min()} bis {future_index.max()} ({len(future_index)} Schritte).")
    
    # 2. Zeit-Features (Monat, Wochentag, Stunde) generieren
    df_future['Monat'] = df_future.index.month
    df_future['Wochentag'] = df_future.index.dayofweek # Montag=0, Sonntag=6
    df_future['Stunde (Lokal)'] = df_future.index.hour
    df_future['IstArbeitstag'] = df_future.index.dayofweek.isin(range(0, 5)).astype(int)
    df_future['IstSonntag'] = (df_future.index.dayofweek == 6).astype(int)
    df_future['Quartal'] = df_future.index.quarter
    
    # 3. Zyklische Kodierung anwenden
    df_future = apply_cyclic_encoding(df_future, 'Monat', 12)
    df_future = apply_cyclic_encoding(df_future, 'Wochentag', 7)
    df_future = apply_cyclic_encoding(df_future, 'Stunde (Lokal)', 24)

    # 4. Lag-Features (Lag_24h) generieren
    # Lag_24h wird berechnet, indem der historische Verbrauch um 24h verschoben wird
    # (Wir nehmen an, dass die Last von vor 24h die beste Näherung ist)
    
    # Kombiniere historische Last und zukünftige Vorhersagen für die Lag-Berechnung
    # Da wir hier nur den 1-Schritt-Forecast machen, benötigen wir nur die letzten historischen Daten.
    # Bei einem iterativen Forecast müsste die Vorhersage zurückgeführt werden.
    
    # Für Lag_24h benötigen wir die historischen Werte bis zum Start des Forecasts.
    required_lag_index = future_index - timedelta(hours=24)
    
    # Werte des Stromverbrauchs 24 Stunden zuvor
    lag_24h_values = df_hist[Config.TARGET_COL].reindex(required_lag_index)
    df_future['Lag_24h'] = lag_24h_values.values

    # 5. Wetter-Features (Persistence Annahme)
    print("  - Fülle Wetter-Features mit Wiederholung des letzten Tagesverlaufs (24h-Muster)")

    lag_features = [f for f in features if f.endswith('_lag15') or f.endswith('_lag30')]

    # 1. Definiere den letzten vollen Tag der historischen Daten
    last_hist_date = df_hist.index.max().normalize() # Der Tag, an dem der Forecast beginnt (z.B. 2025-09-30 00:00:00)

    # Der letzte volle Tag ist 24h vor diesem Datum
    start_of_last_full_day = last_hist_date - timedelta(days=1)
    end_of_last_full_day = last_hist_date - timedelta(minutes=Config.TIME_STEP_MINUTES)

    # 2. Extrahiere den Muster-Tag (24h)
    df_pattern = df_hist.loc[start_of_last_full_day:end_of_last_full_day, lag_features].copy()

    # 3. Erstelle ein zyklisches Muster aus diesem 24h-Tag
    pattern_values = df_pattern.values
    num_steps_to_forecast = len(df_future)

    # Wiederhole den 24h-Pattern so oft wie nötig
    # Beispiel: Für 7 Tage brauchen wir 7 Wiederholungen des 24h-Patterns.
    num_repeats = int(np.ceil(num_steps_to_forecast / len(df_pattern)))
    repeated_pattern = np.tile(pattern_values, (num_repeats, 1))

    # 4. Trimme und Fülle den zukünftigen DataFrame
    df_future[lag_features] = repeated_pattern[:num_steps_to_forecast, :]

    print(f"  - Wetter-Verlauf von {start_of_last_full_day.date()} bis {end_of_last_full_day.date()} wiederholt.")

    # 6. Finalisierung und Konsistenz-Check
    # Stelle sicher, dass der zukünftige DataFrame exakt die Spalten der trainierten Features enthält
    X_future = df_future[features].copy()
    
    print(f"\nÜberprüfung auf {X_future.isna().sum().sum()} NaN-Werte...")

    # Robustes Imputing (ersetzt die alte Zeile 169):
    # a) Backward Fill (für NaN am Anfang, z.B. bei Lag-Features)
    X_future.fillna(method='bfill', inplace=True)
    # b) Forward Fill (für NaN am Ende oder in der Mitte)
    X_future.fillna(method='ffill', inplace=True) 

    # c) Falls immer noch NaN vorhanden (z.B. wenn die gesamte Spalte NaN ist, was unwahrscheinlich ist)
    # Fülle die verbleibenden NaN mit 0 oder einem Durchschnittswert des Trainings-Sets (hier 0 als Notlösung)
    if X_future.isna().any().any():
        X_future.fillna(0, inplace=True) 
        print("WARNUNG: Verbleibende NaN-Werte wurden auf 0 gesetzt.")


    if X_future.isna().any().any():
        # DIESER CHECK sollte jetzt FALSE sein, falls doch, gibt es ein ernstes Datenproblem
        print("FEHLER: Es sind immer noch NaN-Werte im Feature-Set vorhanden!")
        print(X_future.isna().sum()[X_future.isna().sum() > 0]) # Zeige, welche Spalten noch NaN haben

    print(f"X_future erfolgreich erstellt mit {X_future.shape[0]} Schritten und {X_future.shape[1]} Features.")
    
    return X_future

# ============================================================
# MAIN FORECAST LOGIC
# ============================================================

def run_prediction():
    print("=" * 70)
    print("START: ZUKUNFTS-PROGNOSE V11")
    print("=" * 70)
    
    # 1. Modell und Daten laden
    df_hist, model, features = load_historical_data_and_model()
    
    # 2. Zukunfts-Daten vorbereiten
    X_future = prepare_future_data(df_hist, features)
    
    # 3. Prognose erstellen
    print("\nErstelle Prognose...")
    y_pred_future = model.predict(X_future)
    
    # 4. Ergebnisse zusammenführen und speichern
    df_forecast = pd.DataFrame({
        'Datum/Zeit': X_future.index,
        'Prognose_kWh': y_pred_future
    }).set_index('Datum/Zeit')
    
    # Sicherstellen, dass die Prognosewerte positiv sind
    df_forecast['Prognose_kWh'] = df_forecast['Prognose_kWh'].clip(lower=0)
    
    # Speichern der Prognose
    df_forecast.to_csv(Config.PREDICTION_OUTPUT_FILE, sep=';', float_format='%.2f')
    
    print("\n" + "=" * 70)
    print("PROGNOSE ERFOLGREICH ABGESCHLOSSEN")
    print(f"Prognosezeitraum: {df_forecast.index.min()} bis {df_forecast.index.max()}")
    print(f"Ergebnisse gespeichert in: {Config.PREDICTION_OUTPUT_FILE}")
    print("=" * 70)
    
    # Optional: Erste Zeilen der Prognose anzeigen
    print("\nErste 5 Prognosewerte:")
    print(df_forecast.head())

if __name__ == "__main__":
    run_prediction()