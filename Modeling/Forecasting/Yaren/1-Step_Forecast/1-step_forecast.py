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
import matplotlib.pyplot as plt
import joblib

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
START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")

# Ende: nur bis Ende 2024, weil Wetter nur bis dann geht
END_TS = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

# Zeitraum filtern
df = df.loc[START_TS:END_TS].copy()

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
DROP_COLS = [
    "Datum (Lokal)", "Zeit (Lokal)",
    "Monat", "Wochentag", "Stunde (Lokal)",
    "Tag des Jahres","Quartal"
]

EXPERIMENT_NAME = "baseline_full"

USE_WEATHER = True
USE_LAGS = True
USE_CALENDAR = True

EXCLUDE_WEATHER = [
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in km/h_lag15',
    'BÃ¶enspitze (3-SekundenbÃ¶e); Maximum in m/s_lag15',
    'BÃ¶enspitze (SekundenbÃ¶e); Maximum in km/h_lag15',
    'Luftdruck reduziert auf Meeresniveau (QFF)_lag15',
    'Luftdruck reduziert auf Meeresniveau mit Standardatmosphäre (QNH)_lag15',
    'Luftrdruck auf Barometerhöhe_lag15',
    'Lufttemperatur 2 m ü. Gras_lag15',
    'Lufttemperatur Bodenoberfläche_lag15',
    'Windgeschwindigkeit vektoriell_lag15',
    'Windgeschwindigkeit; Zehnminutenmittel in km/h_lag15',
    'Windrichtung; Zehnminutenmittel_lag15',
    'relative Luftfeuchtigkeit_lag15'
]

EXCLUDE_LAGS = [
    "Grundversorgte Kunden_Lag_15min",
    "Freie Kunden_Lag_15min",
    "Lag_15min",
    "Lag_30min",
    "Diff_15min"
]

# --------------------------------------
# Feature-Gruppen
# --------------------------------------
for d in (train_df, test_df):
    d["Monat_sin"] = np.sin(2 * np.pi * d["Monat"] / 12)
    d["Monat_cos"] = np.cos(2 * np.pi * d["Monat"] / 12)

    d["Wochentag_sin"] = np.sin(2 * np.pi * d["Wochentag"] / 7)
    d["Wochentag_cos"] = np.cos(2 * np.pi * d["Wochentag"] / 7)

    d["Stunde_sin"] = np.sin(2 * np.pi * d["Stunde (Lokal)"] / 24)
    d["Stunde_cos"] = np.cos(2 * np.pi * d["Stunde (Lokal)"] / 24)

    d["TagJahr_sin"] = np.sin(2 * np.pi * d["Tag des Jahres"] / 365)
    d["TagJahr_cos"] = np.cos(2 * np.pi * d["Tag des Jahres"] / 365)


CALENDAR_FEATURES = [
    "Monat_sin", "Monat_cos",
    "Wochentag_sin", "Wochentag_cos",
    "Stunde_sin", "Stunde_cos",
    "TagJahr_sin", "TagJahr_cos", "Woche des Jahres",
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
CATEGORICAL_FEATURES = []
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
    #    n_estimators=150,
    #    max_depth=15,
    #    n_jobs=-1
    #),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        n_jobs=-1
    )
}

results = {}

numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

transformers = [("num", "passthrough", NUMERIC_FEATURES)]
if len(CATEGORICAL_FEATURES) > 0:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES))

preprocessor = ColumnTransformer(transformers=transformers)

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

# ===========================
# 1-Step-Modell: 24h-Ausschnitt fuer 15.11.2024
# ===========================

# y_test ist eine Series mit Index = Zeitstempel
# y_pred_test ist ein np.array -> wir bauen eine Series mit gleichem Index
y_pred_test_series = pd.Series(y_pred_test, index=y_test.index)

target_start = pd.Timestamp("2024-11-15 00:00:00", tz="UTC")
target_end   = target_start + pd.Timedelta("24h") - pd.Timedelta("15min")  # 96 Punkte

mask = (y_test.index >= target_start) & (y_test.index <= target_end)

y_true_15 = y_test.loc[mask]
y_pred_15 = y_pred_test_series.loc[mask]

print("1-Step 15.11. | Punkte:", len(y_true_15))
print("Zeitraum:", y_true_15.index.min(), "→", y_true_15.index.max())

# Metriken nur fuer diesen Tag
mae_15  = mean_absolute_error(y_true_15.values, y_pred_15.values)
rmse_15 = np.sqrt(mean_squared_error(y_true_15.values, y_pred_15.values))
r2_15   = r2_score(y_true_15.values, y_pred_15.values)

# in lokale Zeit fuer Plot
idx_local = y_true_15.index.tz_convert("Europe/Zurich")

plt.figure(figsize=(12, 4))
plt.plot(idx_local, y_true_15.values, label="True 1-Step (15.11.)")
plt.plot(idx_local, y_pred_15.values, label="Pred 1-Step (15.11.)")
plt.title(f"1-Step Forecast (klassisch) fuer 15.11.2024\nMAE={mae_15:.0f}, RMSE={rmse_15:.0f}, R²={r2_15:.3f}")
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== 1-Step Forecast 15.11.2024 ===")
print(f"MAE  : {mae_15:.2f}")
print(f"RMSE : {rmse_15:.2f}")
print(f"R2   : {r2_15:.4f}")


# === 24H - MULTI - Step FORECAST ===
def forecast_multistep(pipe, df_history, df_full, steps=96):
    """
    pipe      : trainierte Pipeline (z.B. XGBoost)
    df_history: Vergangenheit mit Stromverbrauch (und optional weiteren Spalten)
    df_full   : kompletter Feature-DataFrame mit Wetter-Lag-Spalten
    steps     : Anzahl 15-Minuten-Schritte in die Zukunft (96 = 24h)

    Nutzt:
      - Kalenderfeatures aus next_ts
      - Lag-Features aus df_temp (nur Stromverbrauch, Kunden, Diff)
      - Wetterfeatures (_lag15) aus df_full, falls vorhanden
    """
    df_temp = df_history.copy().sort_index()
    preds = []

    for _ in range(steps):
        last_ts = df_temp.index[-1]
        next_ts = last_ts + pd.Timedelta("15min")

        row = {}

        # --- Kalender-Features ---
        if "Monat" in FEATURES:
            row["Monat"] = next_ts.month
        if "Wochentag" in FEATURES:
            row["Wochentag"] = next_ts.weekday()
        if "Quartal" in FEATURES:
            row["Quartal"] = (next_ts.month - 1) // 3 + 1
        if "Woche des Jahres" in FEATURES:
            row["Woche des Jahres"] = next_ts.isocalendar().week
        if "Stunde (Lokal)" in FEATURES:
            row["Stunde (Lokal)"] = next_ts.tz_convert("Europe/Zurich").hour
        if "IstArbeitstag" in FEATURES:
            w = next_ts.weekday()
            row["IstArbeitstag"] = int(w < 5)
        if "IstSonntag" in FEATURES:
            row["IstSonntag"] = int(next_ts.weekday() == 6)

        # --- Lag-Features (Stromverbrauch) ---
        if "Lag_15min" in FEATURES:
            row["Lag_15min"] = df_temp.iloc[-1]["Stromverbrauch"]
        if "Lag_30min" in FEATURES and len(df_temp) >= 2:
            row["Lag_30min"] = df_temp.iloc[-2]["Stromverbrauch"]
        if "Lag_1h" in FEATURES and len(df_temp) >= 4:
            row["Lag_1h"] = df_temp.iloc[-4]["Stromverbrauch"]
        if "Lag_24h" in FEATURES and len(df_temp) >= 96:
            row["Lag_24h"] = df_temp.iloc[-96]["Stromverbrauch"]

        if "Diff_15min" in FEATURES and len(df_temp) >= 2:
            v1 = df_temp.iloc[-1]["Stromverbrauch"]
            v2 = df_temp.iloc[-2]["Stromverbrauch"]
            row["Diff_15min"] = v1 - v2

        # --- Kunden-Lags (falls benutzt) ---
        if "Grundversorgte Kunden_Lag_15min" in FEATURES and "Grundversorgte Kunden_Lag_15min" in df_temp.columns:
            row["Grundversorgte Kunden_Lag_15min"] = df_temp.iloc[-1]["Grundversorgte Kunden_Lag_15min"]
        if "Freie Kunden_Lag_15min" in FEATURES and "Freie Kunden_Lag_15min" in df_temp.columns:
            row["Freie Kunden_Lag_15min"] = df_temp.iloc[-1]["Freie Kunden_Lag_15min"]

        # --- Wetter-Lag-Features aus df_full ---
        # Wichtig: df_full enthält bereits *_lag15, also einfach Zeile bei next_ts nehmen
        if USE_WEATHER and len(WEATHER_FEATURES) > 0:
            if next_ts in df_full.index:
                for feat in WEATHER_FEATURES:
                    if feat in FEATURES:
                        row[feat] = df_full.loc[next_ts, feat]
            else:
                # Falls wir ueber das bekannte Datenende hinausgehen:
                # letzte bekannte Werte einfrieren (besser als nichts)
                last_known_ts = df_full.index[df_full.index < next_ts].max()
                for feat in WEATHER_FEATURES:
                    if feat in FEATURES:
                        row[feat] = df_full.loc[last_known_ts, feat]

        # DataFrame für genau die Spalten, die die Pipeline kennt
        X = pd.DataFrame([row], index=[next_ts])
        X = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        # Vorhersage
        y_hat = pipe.predict(X)[0]
        preds.append((next_ts, y_hat))

        # neue Zeile in df_temp: wir brauchen nur Stromverbrauch (und optional Kundenlags)
        new_row = {"Stromverbrauch": y_hat}
        if "Grundversorgte Kunden_Lag_15min" in df_temp.columns:
            new_row["Grundversorgte Kunden_Lag_15min"] = df_temp.iloc[-1].get("Grundversorgte Kunden_Lag_15min", np.nan)
        if "Freie Kunden_Lag_15min" in df_temp.columns:
            new_row["Freie Kunden_Lag_15min"] = df_temp.iloc[-1].get("Freie Kunden_Lag_15min", np.nan)

        df_temp.loc[next_ts] = new_row

    forecast_df = pd.DataFrame(preds, columns=["ts", "forecast"]).set_index("ts")
    return forecast_df


# Für den Plot brauchst du das letzte trainierte Modell (XGBoost)
pipe = pipe  # Pipeline aus der letzten Schleife

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

# ===========================
# 24h Multi-Step Forecast für 15.11.2024 (rekursiv)
# ===========================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ziel: 24h-Lastgang am 15.11.2024 (UTC) vorhersagen
target_day_end = pd.Timestamp("2024-11-15 23:45:00", tz="UTC")

if target_day_end not in df.index:
    print("Fehler: Zielzeitpunkt nicht im Datensatz:", target_day_end)
else:
    # Wir wollen 24h bis target_day_end forecasten
    last_ts = target_day_end

    # Ende der Historie = last_ts - 24h (also 14.11. 23:45)
    history_end = last_ts - pd.Timedelta("24h")
    history_start = history_end - pd.Timedelta("7D")  # 7 Tage Historie

    history = df.loc[history_start:history_end].copy()

    print("History:", history.index.min(), "→", history.index.max())
    print("Ziel-Tag-Ende:", last_ts)

    horizon = 96  # 24h in 15min-Schritten
    fc_24h = forecast_multistep(pipe, history, df_full=df, steps=horizon)

    print("Forecast:", fc_24h.index.min(), "→", fc_24h.index.max())

    # Nur Timestamps verwenden, die wirklich im df existieren
    common_idx = fc_24h.index.intersection(df.index)

    if len(common_idx) == 0:
        print("Keine True-Werte für 24h-Vergleich vorhanden (Forecast ausserhalb Datenbereich).")
    else:
        # True + Pred für gleiche Zeitpunkte
        y_true_24h = df.loc[common_idx, TARGET_COL]
        y_pred_24h = fc_24h.loc[common_idx, "forecast"]

        # lokale Zeit für Plot
        idx_local = y_true_24h.index.tz_convert("Europe/Zurich")

        plt.figure(figsize=(12, 4))
        plt.plot(idx_local, y_true_24h.values, label="True 24h")
        plt.plot(idx_local, y_pred_24h.values, label="Forecast 24h (recursive)")
        plt.legend()
        plt.title("24h Multi-Step Forecast (rekursiv) für 15.11.2024")
        plt.tight_layout()
        plt.show()

        # 24h-Metriken berechnen
        y_true_vals = y_true_24h.values
        y_pred_vals = y_pred_24h.values

        mae_24h = mean_absolute_error(y_true_vals, y_pred_vals)
        rmse_24h = np.sqrt(mean_squared_error(y_true_vals, y_pred_vals))
        r2_24h = r2_score(y_true_vals, y_pred_vals)

        # MAPE mit Schutz vor 0
        y_true_safe = np.where(y_true_vals == 0, 1e-6, y_true_vals)
        mape_24h = np.mean(np.abs((y_true_vals - y_pred_vals) / y_true_safe)) * 100

        print("\n=== 24h Multi-Step Forecast (rekursiv) für 15.11.2024 ===")
        print(f"MAE 24h  : {mae_24h:.2f}")
        print(f"RMSE 24h : {rmse_24h:.2f}")
        print(f"R2 24h   : {r2_24h:.4f}")
        print(f"MAPE 24h : {mape_24h:.2f} %")


MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_1step_pipeline.joblib")

joblib.dump(pipe, MODEL_PATH)

print("Modell gespeichert unter:", MODEL_PATH)
