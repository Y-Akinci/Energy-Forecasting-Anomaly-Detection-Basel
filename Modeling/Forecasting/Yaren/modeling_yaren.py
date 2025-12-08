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

INPUT_CSV = os.path.join(DATA_DIR, "processed_merged_features.csv")

df = pd.read_csv(
    INPUT_CSV,
    sep=";",
    encoding="latin1",
    parse_dates=["Start der Messung (UTC)"]
)

df.set_index("Start der Messung (UTC)", inplace=True)

# WICHTIG: Nur lokalizen, wenn noch keine TZ dran hängt
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

print("Index-TZ:", df.index.tz)
print(df.head())

# fester Startzeitpunkt, ab dem beide Kundenspalten vorhanden sind
START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")

# Ende: nur bis Ende 2024, weil Wetter nur bis dann geht
END_TS = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

# Zeitraum filtern
df = df.loc[START_TS:END_TS].copy()

# optional: sicherstellen, dass Kunden nicht null sind
df = df.dropna(subset=["Grundversorgte Kunden_Lag_15min", "Freie Kunden_Lag_15min"])

print("Zeitraum nach Filter:", df.index.min(), "→", df.index.max())
print("Anzahl Zeilen:", len(df))

# 70 / 30 Split nach Reihenfolge (zeitlich korrekt)

n = len(df)
split_idx = int(n * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print("Train-Zeilen:", len(train_df), "| Zeitraum:",
      train_df.index.min(), "→", train_df.index.max())
print("Test-Zeilen :", len(test_df), "| Zeitraum:",
      test_df.index.min(), "→", test_df.index.max())


EXPERIMENT_NAME = "baseline_full"

USE_WEATHER = True
USE_LAGS = True
USE_CALENDAR = True

EXCLUDE_WEATHER = [
    
]

TARGET_COL = "Stromverbrauch" 
DROP_COLS = ["Datum (Lokal)", "Zeit (Lokal)"]

CATEGORICAL_FEATURES = []
for col in ["Monat", "Wochentag", "Quartal", "Woche des Jahres", "Stunde (Lokal)", "IstArbeitstag", "IstSonntag"]:
    if col in train_df.columns:
        CATEGORICAL_FEATURES.append(col)

NUMERIC_FEATURES = [
    c for c in train_df.columns
    if c not in CATEGORICAL_FEATURES + DROP_COLS + [TARGET_COL]
]
print("\nKategorische Features:", CATEGORICAL_FEATURES)
print("Numerische Features   :", NUMERIC_FEATURES[:10], "...")
print("Anzahl numärische Features:", len(NUMERIC_FEATURES))
print("Anzahl kategorische Features:", len(CATEGORICAL_FEATURES))

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

numeric_transformer = "passthrough" # numärische Featrues durchreichen

categorial_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
     ("cat", categorial_transformer, CATEGORICAL_FEATURES),
])

models = {
    "LinearRegression": LinearRegression(),
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

for name, model_obj in models.items():
    print(f"\n Training {name} ")

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model_obj)
        ]
    )


    # Training
    pipe.fit(X_train, y_train)

    # Vorhersage auf TRAIN
    y_pred_train = pipe.predict(X_train)

    # Vorhersage auf TEST
    y_pred_test = pipe.predict(X_test)

    # --- Metriken Trainingsdaten ---
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    mape_train = np.mean(
        np.abs((y_train - y_pred_train) / np.where(y_train == 0, 1e-6, y_train))
    ) * 100

    # --- Metriken Testdaten ---
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(
        np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-6, y_test))
    ) * 100
    
    # Alles in results speichern
    results[name] = {
        "MAE_train": mae_train,
        "RMSE_train": rmse_train,
        "R2_train": r2_train,
        "MAPE_train": mape_train,
        "MAE_test": mae_test,
        "RMSE_test": rmse_test,
        "R2_test": r2_test,
        "MAPE_test": mape_test,
    }

    print(f"Train: MAE={mae_train:.1f}, RMSE={rmse_train:.1f}, R2={r2_train:.5f}, MAPE={mape_train:.2f}%")
    print(f"Test : MAE={mae_test:.1f}, RMSE={rmse_test:.1f}, R2={r2_test:.5f}, MAPE={mape_test:.2f}%")