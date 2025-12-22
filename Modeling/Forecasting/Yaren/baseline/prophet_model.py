import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prophet import (je nach Installation)
# pip install prophet
from prophet import Prophet


# =========================================================
# SETTINGS (nur hier anfassen)
# =========================================================
CSV_FILENAME = "processed_merged_features.csv"
DATE_COL = "Start der Messung (UTC)"
TARGET_COL = "Stromverbrauch"

START_TS = pd.Timestamp("2020-08-31 22:00:00", tz="UTC")
END_TS   = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

AGG = "sum"       # "sum" fuer Tagesenergie, "mean" fuer Tagesmittel
TRAIN_RATIO = 0.7 # wie bei dir


# =========================================================
# Pfade bauen (gleiches Muster wie deine Scripts)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
DATA_DIR = os.path.join(ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, CSV_FILENAME)

print("INPUT_CSV:", INPUT_CSV)


# =========================================================
# Daten laden + Index
# =========================================================
df = pd.read_csv(
    INPUT_CSV,
    sep=";",
    encoding="latin1",
    parse_dates=[DATE_COL]
)

df.set_index(DATE_COL, inplace=True)

if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

df = df.loc[START_TS:END_TS].copy()
print("Zeitraum:", df.index.min(), "→", df.index.max(), "| Zeilen:", len(df))


# =========================================================
# Daily Aggregation
# =========================================================
# Optional: auf Lokalzeit aggregieren (macht bei Tagesaggregation oft mehr Sinn)
df_local = df.copy()
df_local.index = df_local.index.tz_convert("Europe/Zurich")

# Nur volle Tage behalten: 96 Werte pro Tag (15min Raster)
counts = df_local[TARGET_COL].resample("D").count()

if AGG == "sum":
    daily = df_local[TARGET_COL].resample("D").sum()
elif AGG == "mean":
    daily = df_local[TARGET_COL].resample("D").mean()
else:
    raise ValueError("AGG muss 'sum' oder 'mean' sein")

# Filter: nur komplette Tage
daily = daily[counts == 96].dropna()
print("Letzter Daily-Tag:", daily.index.max(), "| count:", counts.loc[daily.index.max()])
print("Daily rows:", len(daily), "| von", daily.index.min(), "bis", daily.index.max())


# =========================================================
# Prophet Format: ds / y
# =========================================================
prophet_df = daily.reset_index()
prophet_df.columns = ["ds", "y"]
# Prophet erwartet naive timestamps oder timezone-naive. Wir machen sie naive.
prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

# Zeitlicher Split
n = len(prophet_df)
split_idx = int(n * TRAIN_RATIO)

train_df = prophet_df.iloc[:split_idx].copy()
test_df  = prophet_df.iloc[split_idx:].copy()

print("Train days:", len(train_df), "|", train_df["ds"].min(), "→", train_df["ds"].max())
print("Test  days:", len(test_df),  "|", test_df["ds"].min(),  "→", test_df["ds"].max())


# =========================================================
# Prophet Modell
# =========================================================
m = Prophet(
    daily_seasonality=False,   # daily seasonality macht bei daily daten keinen Sinn
    weekly_seasonality=True,
    yearly_seasonality=True,
)

m.fit(train_df)

# Forecast exakt fuer Test-Zeitraum
future = test_df[["ds"]].copy()
forecast = m.predict(future)

y_true = test_df["y"].values
y_pred = forecast["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n=== Prophet Daily Baseline ===")
print(f"AGG={AGG} | MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.4f}")


# =========================================================
# Plot
# =========================================================
plt.figure(figsize=(12, 4))
plt.plot(test_df["ds"], y_true, label="True (Daily)")
plt.plot(test_df["ds"], y_pred, label="Prophet yhat")
plt.title(f"Prophet Daily Baseline (AGG={AGG}) | MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
plt.legend()
plt.tight_layout()
plt.show()

# Komponentenplot (optional)
# m.plot_components(forecast)
# plt.show()
