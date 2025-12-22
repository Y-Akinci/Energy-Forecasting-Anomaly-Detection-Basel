import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# =======================================
# Pfade / Daten laden
# =======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_DIR = os.path.join(ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, "processed_merged_features.csv")

df = pd.read_csv(
    INPUT_CSV,
    sep=";",
    encoding="latin1",
    parse_dates=["Start der Messung (UTC)"]
)

df.set_index("Start der Messung (UTC)", inplace=True)

if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")
df = df.loc[START_TS:END_TS].copy()

print("Index-TZ:", df.index.tz)
print("Zeitraum:", df.index.min(), "→", df.index.max())
print("Anzahl Zeilen:", len(df))


# =======================================
# Feature-Setup (wie dein rekursiver Teil)
# =======================================

TARGET_COL = "Stromverbrauch"
DROP_COLS = ["Datum (Lokal)", "Zeit (Lokal)"]

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

# Wetter: alles *_lag15, ausser Lags
WEATHER_FEATURES_ALL = [
    c for c in df.columns
    if c.endswith("_lag15") and c not in LAG_FEATURES
]

WEATHER_FEATURES = [
    c for c in WEATHER_FEATURES_ALL
    if c not in EXCLUDE_WEATHER
]

LAG_FEATURES = [c for c in LAG_FEATURES if c not in EXCLUDE_LAGS]

FEATURES = []
if USE_CALENDAR:
    FEATURES += CALENDAR_FEATURES
if USE_LAGS:
    FEATURES += LAG_FEATURES
if USE_WEATHER:
    FEATURES += WEATHER_FEATURES

FEATURES = [f for f in FEATURES if f in df.columns]

CATEGORICAL_FEATURES = [c for c in CALENDAR_FEATURES if c in FEATURES]
NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]

print("\nAktive Features:", len(FEATURES))
print("Numerische:", len(NUMERIC_FEATURES))
print("Kategorische:", len(CATEGORICAL_FEATURES))

# Drop Display-Spalten
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])


# =======================================
# Train / Test Split (zeitlich)
# =======================================

n = len(df)
split_idx = int(n * 0.7)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print("\nTrain:", train_df.index.min(), "→", train_df.index.max(), "|", len(train_df))
print("Test :", test_df.index.min(), "→", test_df.index.max(), "|", len(test_df))

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET_COL].astype("float64")
y_test = test_df[TARGET_COL].astype("float64")


# =======================================
# Pipeline (wie bei dir: OneHot fuer Kalender)
# =======================================

transformers = [
    ("num", "passthrough", NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
]
preprocessor = ColumnTransformer(transformers=transformers)

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        n_jobs=-1
    ))
])

print("\n=== Training XGBoost ===")
pipe.fit(X_train, y_train)


# =======================================
# Rekursiver 24h Multi-Step Forecast
# =======================================

def forecast_multistep(pipe, df_history, df_full, steps=96):
    df_temp = df_history.copy().sort_index()
    preds = []

    for _ in range(steps):
        last_ts = df_temp.index[-1]
        next_ts = last_ts + pd.Timedelta("15min")

        row = {}

        # Kalender
        if "Monat" in FEATURES:
            row["Monat"] = next_ts.month
        if "Wochentag" in FEATURES:
            row["Wochentag"] = next_ts.weekday()
        if "Quartal" in FEATURES:
            row["Quartal"] = (next_ts.month - 1) // 3 + 1
        if "Woche des Jahres" in FEATURES:
            row["Woche des Jahres"] = int(next_ts.isocalendar().week)
        if "Stunde (Lokal)" in FEATURES:
            row["Stunde (Lokal)"] = next_ts.tz_convert("Europe/Zurich").hour
        if "IstArbeitstag" in FEATURES:
            row["IstArbeitstag"] = int(next_ts.weekday() < 5)
        if "IstSonntag" in FEATURES:
            row["IstSonntag"] = int(next_ts.weekday() == 6)

        # Lags (Target)
        if "Lag_15min" in FEATURES:
            row["Lag_15min"] = df_temp.iloc[-1][TARGET_COL]
        if "Lag_30min" in FEATURES and len(df_temp) >= 2:
            row["Lag_30min"] = df_temp.iloc[-2][TARGET_COL]
        if "Lag_1h" in FEATURES and len(df_temp) >= 4:
            row["Lag_1h"] = df_temp.iloc[-4][TARGET_COL]
        if "Lag_24h" in FEATURES and len(df_temp) >= 96:
            row["Lag_24h"] = df_temp.iloc[-96][TARGET_COL]

        if "Diff_15min" in FEATURES and len(df_temp) >= 2:
            row["Diff_15min"] = df_temp.iloc[-1][TARGET_COL] - df_temp.iloc[-2][TARGET_COL]

        # Kunden-Lags (falls vorhanden)
        if "Grundversorgte Kunden_Lag_15min" in FEATURES and "Grundversorgte Kunden_Lag_15min" in df_temp.columns:
            row["Grundversorgte Kunden_Lag_15min"] = df_temp.iloc[-1]["Grundversorgte Kunden_Lag_15min"]
        if "Freie Kunden_Lag_15min" in FEATURES and "Freie Kunden_Lag_15min" in df_temp.columns:
            row["Freie Kunden_Lag_15min"] = df_temp.iloc[-1]["Freie Kunden_Lag_15min"]

        # Wetter
        if USE_WEATHER and len(WEATHER_FEATURES) > 0:
            if next_ts in df_full.index:
                for feat in WEATHER_FEATURES:
                    if feat in FEATURES:
                        row[feat] = df_full.loc[next_ts, feat]
            else:
                last_known_ts = df_full.index[df_full.index < next_ts].max()
                for feat in WEATHER_FEATURES:
                    if feat in FEATURES:
                        row[feat] = df_full.loc[last_known_ts, feat]

        X = pd.DataFrame([row], index=[next_ts])

        # Safety: fehlende Spalten fuellen
        for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES):
            if c not in X.columns:
                X[c] = np.nan

        X = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        y_hat = pipe.predict(X)[0]
        preds.append((next_ts, y_hat))

        df_temp.loc[next_ts] = {TARGET_COL: y_hat}

    return pd.DataFrame(preds, columns=["ts", "forecast"]).set_index("ts")


# =======================================
# Tagesmetriken fuer 24h Multi-Step (rekursiv)
# =======================================

def metrics_24h_block(y_true_vals, y_pred_vals):
    mae = mean_absolute_error(y_true_vals, y_pred_vals)
    rmse = np.sqrt(mean_squared_error(y_true_vals, y_pred_vals))
    r2 = r2_score(y_true_vals, y_pred_vals)
    y_true_safe = np.where(y_true_vals == 0, 1e-6, y_true_vals)
    mape = np.mean(np.abs((y_true_vals - y_pred_vals) / y_true_safe)) * 100
    return mae, rmse, r2, mape

def daily_multistep_metrics(df_full, period_start, period_end, label):
    """
    period_start/period_end sind UTC timestamps (inklusive).
    Pro Tag: forecast 24h (96 steps) und Metriken speichern.
    """
    # Tage definieren als UTC-Tage 00:00 → 23:45
    days = pd.date_range(
        start=period_start.normalize(),
        end=period_end.normalize(),
        freq="D",
        tz="UTC"
    )

    rows = []
    for day in days:
        target_start = day
        target_end = day + pd.Timedelta("24h") - pd.Timedelta("15min")

        # Wir brauchen True-Werte fuer den ganzen Tag
        if target_end not in df_full.index:
            continue

        # Historie: Ende = 24h vor target_end (also day-1 23:45), Start = 7 Tage davor
        history_end = target_end - pd.Timedelta("24h")
        history_start = history_end - pd.Timedelta("7D")

        # Historie muss existieren
        if history_start not in df_full.index or history_end not in df_full.index:
            continue

        history = df_full.loc[history_start:history_end].copy()

        # Forecast
        fc = forecast_multistep(pipe, history, df_full=df_full, steps=96)

        common_idx = fc.index.intersection(df_full.index)
        if len(common_idx) != 96:
            continue

        y_true_24h = df_full.loc[common_idx, TARGET_COL].values
        y_pred_24h = fc.loc[common_idx, "forecast"].values

        mae, rmse, r2, mape = metrics_24h_block(y_true_24h, y_pred_24h)

        rows.append([
            day.date(), mae, rmse, r2, mape
        ])

    out = pd.DataFrame(rows, columns=["Tag", "MAE_24h", "RMSE_24h", "R2_24h", "MAPE_24h_%"])
    print(f"\n=== Tagesmetriken 24h Multi-Step rekursiv ({label}) ===")
    print(out.to_string(index=False))
    return out


# Train- und Testperiode fuer Tagesliste
train_start = train_df.index.min()
train_end = train_df.index.max()

test_start = test_df.index.min()
test_end = test_df.index.max()

daily_train_24h = daily_multistep_metrics(df, train_start, train_end, "Train")
daily_test_24h = daily_multistep_metrics(df, test_start, test_end, "Test")


# =======================================
# Plot + Metriken fuer 15.11.2024 (24h rekursiv)
# =======================================

day_plot = pd.Timestamp("2024-11-15 00:00:00", tz="UTC")
target_end_plot = day_plot + pd.Timedelta("24h") - pd.Timedelta("15min")

if target_end_plot not in df.index:
    raise ValueError("15.11.2024 nicht voll im Datensatz vorhanden.")

history_end = target_end_plot - pd.Timedelta("24h")
history_start = history_end - pd.Timedelta("7D")
history = df.loc[history_start:history_end].copy()

fc_24h = forecast_multistep(pipe, history, df_full=df, steps=96)

common_idx = fc_24h.index.intersection(df.index)
y_true_24h = df.loc[common_idx, TARGET_COL]
y_pred_24h = fc_24h.loc[common_idx, "forecast"]

mae_24h, rmse_24h, r2_24h, mape_24h = metrics_24h_block(y_true_24h.values, y_pred_24h.values)

idx_local = y_true_24h.index.tz_convert("Europe/Zurich")

plt.figure(figsize=(12, 4))
plt.plot(idx_local, y_true_24h.values, label="True 24h")
plt.plot(idx_local, y_pred_24h.values, label="Forecast 24h (rekursiv)")
plt.legend()
plt.title(f"24h Multi-Step Forecast (rekursiv) fuer 15.11.2024\nMAE={mae_24h:.0f}, RMSE={rmse_24h:.0f}, R2={r2_24h:.3f}, MAPE={mape_24h:.2f}%")
plt.tight_layout()
plt.show()

print("\n=== 15.11.2024 24h Multi-Step rekursiv ===")
print(f"MAE  : {mae_24h:.2f}")
print(f"RMSE : {rmse_24h:.2f}")
print(f"R2   : {r2_24h:.4f}")
print(f"MAPE : {mape_24h:.2f} %")
