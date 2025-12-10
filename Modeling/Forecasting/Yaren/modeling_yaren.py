import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import pandas as pd


# Ordner, in dem dieses modeling_yaren.py liegt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Projekt-Root = 3 Ebenen höher
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
# Data-Ordner
DATA_DIR = os.path.join(ROOT, "data")

# Pfad zur CSV
INPUT_CSV = os.path.join(DATA_DIR, "processed_merged_features.csv")

# CSV einlesen
df = pd.read_csv(
    INPUT_CSV,
    sep=";",
    encoding="latin1",
    parse_dates=["Start der Messung (UTC)"]
)

# UTC-Zeiten in 15min Intervallen werden als Index gesetzt.
df.set_index("Start der Messung (UTC)", inplace=True)

# WICHTIG: Nur lokalizen, wenn noch keine TZ dran hängt
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

print("Index-TZ:", df.index.tz)
print(df.head())

# fester Startzeitpunkt, ab dem beide Kundenspalten vorhanden sind
START_TS = pd.Timestamp("2012-08-31 22:00:00", tz="UTC")

# Ende: nur bis Ende 2024, weil Wetter nur bis dann geht
END_TS = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

# Zeitraum filtern
df = df.loc[START_TS:END_TS].copy()

# optional: sicherstellen, dass Kunden nicht null sind
#df = df.dropna(subset=["Grundversorgte Kunden_Lag_15min", "Freie Kunden_Lag_15min"])

print("Zeitraum nach Filter:", df.index.min(), "→", df.index.max())
print("Anzahl Zeilen:", len(df))

# 70 / 30 Split nach Reihenfolge (zeitlich korrekt)
n = len(df)
split_idx = int(n * 0.7)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print("Train-Zeilen:", len(train_df), "| Zeitraum:",
      train_df.index.min(), "→", train_df.index.max())
print("Test-Zeilen :", len(test_df), "| Zeitraum:",
      test_df.index.min(), "→", test_df.index.max())


TARGET_COL = "Stromverbrauch" 
DROP_COLS = ["Datum (Lokal)", "Zeit (Lokal)"]

EXPERIMENT_NAME = "baseline_full"   # beliebig, du notierst es in markdown

USE_WEATHER = True             # komplette Wetterdaten nutzen?
USE_LAGS = True                # Lag Features nutzen?
USE_CALENDAR = True  

EXCLUDE_WEATHER = [
   

    ]

EXCLUDE_LAGS = ["Grundversorgte Kunden_Lag_15min", "Freie Kunden_Lag_15min"
    
]

# --------------------------------------
# Feature-Gruppen
# --------------------------------------

CALENDAR_FEATURES = [
    "Monat", "Wochentag", "Quartal", 
    "Woche des Jahres", "Stunde (Lokal)",
    "IstArbeitstag", "IstSonntag"
]

LAG_FEATURES = [
    "Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h",
    "Grundversorgte Kunden_Lag_15min",
    "Freie Kunden_Lag_15min",
    "Diff_15min"
]

# Wetter: alles, was mit _lag15 endet
WEATHER_FEATURES_ALL = [
    c for c in df.columns 
    if c.endswith("_lag15") and c not in LAG_FEATURES
]

# einzelne Wetterfeatures deaktivieren
WEATHER_FEATURES = [
    c for c in WEATHER_FEATURES_ALL 
    if c not in EXCLUDE_WEATHER
]


LAG_FEATURES = [
    c for c in LAG_FEATURES
    if c not in EXCLUDE_LAGS
]

# Finales Feature-Set 
FEATURES = []
if USE_CALENDAR:
    FEATURES += CALENDAR_FEATURES
if USE_LAGS:
    FEATURES += LAG_FEATURES
if USE_WEATHER:
    FEATURES += WEATHER_FEATURES

FEATURES = [f for f in FEATURES if f in df.columns]  # Safety

# kategorisch/numerisch trennen
CATEGORICAL_FEATURES = [c for c in CALENDAR_FEATURES if c in FEATURES]
NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]

print("\nAktive Features:", len(FEATURES))
print("Numerische:", len(NUMERIC_FEATURES))
print("Kategorische:", len(CATEGORICAL_FEATURES))



# Drop ueberfluessige Spalten, falls vorhanden
train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])


# Ziel
y_train = train_df[TARGET_COL].astype("float64")
y_test = test_df[TARGET_COL].astype("float64")

# Features
X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


print("X_train Shape:", X_train.shape, " | y_train:", y_train.shape)
print("X_test  Shape:", X_test.shape, " | y_test :", y_test.shape)

MODELS = {
    #"LinearRegression": LinearRegression(),
    #"RandomForest": RandomForestRegressor(
        #n_estimators=150,
        #max_depth=15,
        #n_jobs=-1
    #),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        n_jobs=-1)
}

results = {}

numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

for name, model_obj in MODELS.items():

    print(f"\n=== Training {name} ===")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model_obj)
    ])

    pipe.fit(X_train, y_train)

    # Predict Train + Test
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    metrics = {
        "MAE_train": mean_absolute_error(y_train, y_pred_train),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "R2_train": r2_score(y_train, y_pred_train),

        "MAE_test": mean_absolute_error(y_test, y_pred_test),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R2_test": r2_score(y_test, y_pred_test),
    }

    results[name] = metrics

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

results_df = pd.DataFrame(results).T

print("\n=== RESULTS (Train/Test) ===")
print(results_df[[
    "MAE_train", "RMSE_train", "R2_train",
    "MAE_test", "RMSE_test", "R2_test"
]])


def forecast_multistep(pipe, df_history, steps=96*7):
    df_temp = df_history.copy()
    preds = []

    for _ in range(steps):
        last_ts = df_temp.index[-1]
        next_ts = last_ts + pd.Timedelta("15min")

        # Neue Features bauen: Kalender
        row = {}
        row["Monat"] = next_ts.month
        row["Wochentag"] = next_ts.weekday()
        row["Quartal"] = (next_ts.month-1)//3 + 1
        row["Woche des Jahres"] = next_ts.isocalendar().week
        row["Stunde (Lokal)"] = next_ts.tz_convert("Europe/Zurich").hour
        row["IstArbeitstag"] = int(row["Wochentag"] < 5)
        row["IstSonntag"] = int(row["Wochentag"] == 6)

        # Lags aus df_temp
        row["Lag_15min"] = df_temp.iloc[-1]["Stromverbrauch"]
        row["Lag_30min"] = df_temp.iloc[-2]["Stromverbrauch"]
        row["Lag_1h"] = df_temp.iloc[-4]["Stromverbrauch"]
        row["Lag_24h"] = df_temp.iloc[-96]["Stromverbrauch"]

        # Watte: Wetter? -> DU ENTSCHEIDEST
        # Wenn du Wetter willst, brauchst du Echtzeit- oder Forecast-Daten.
        # Sonst: lass Wetter weg.

        X = pd.DataFrame([row], index=[next_ts])

        y_hat = pipe.predict(X)[0]

        preds.append((next_ts, y_hat))

        df_temp.loc[next_ts, "Stromverbrauch"] = y_hat

    forecast_df = pd.DataFrame(preds, columns=["ts", "forecast"]).set_index("ts")
    return forecast_df



import matplotlib.pyplot as plt

# Für den Plot brauchst du das letzte trainierte Modell:
last_model_name = list(results.keys())[-1]
pipe = pipe   # letztes trainiertes Pipeline-Modell

# Vorhersagen
y_pred_train = pipe.predict(X_train)
y_pred_test  = pipe.predict(X_test)

# Plot: Vergleich Testzeitraum (erste 500 Punkte)
N = 500
plt.figure(figsize=(12,4))
plt.plot(y_test.index[:N], y_test.values[:N], label="True")
plt.plot(y_test.index[:N], y_pred_test[:N], label="Pred")
plt.title("Test: True vs Pred")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred_test, s=3)
plt.xlabel("True")
plt.ylabel("Pred")
plt.title("Scatter Test")
plt.tight_layout()
plt.show()
