import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump  

# ============================================================
# 1) DATEN LADEN
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))  # 2x hoch statt 1x
DATA_DIR = os.path.join(REPO_ROOT, "Data")  # Großes D!
INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")

print("Lade Daten aus:", INPUT_FILE)

# Wenn die Datei nicht gefunden wird, prüfe einen möglichen alternativen Pfad
if not os.path.isfile(INPUT_FILE):
    alt_input = os.path.join(ROOT, "data", "processed_merged_features.csv")
    if os.path.isfile(alt_input):
        print("Hinweis: Datei stattdessen unter Modeling/data gefunden. Verwende alternativen Pfad.")
        INPUT_FILE = alt_input
    else:
        print("Datei nicht gefunden. Überprüfe folgende Pfade:")
        print(" - Erwarteter DATA_DIR:", DATA_DIR, "exists:", os.path.exists(DATA_DIR))
        print(" - Alternatives DATA_DIR:", os.path.join(ROOT, "data"), "exists:", os.path.exists(os.path.join(ROOT, "data")))
        raise FileNotFoundError(
            f"Kann Input-Datei nicht finden: {INPUT_FILE}\nBitte lege 'processed_merged_features.csv' in eines der oben genannten data-Verzeichnisse.")

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    encoding="latin1",  # ← Auch encoding ändern!
    index_col=0,
    parse_dates=True
)
df = df.sort_index()
print("Gesamtzeitraum:", df.index.min(), "→", df.index.max(), " Rows:", len(df))

# ============================================================
# 2) ZEITFENSTER WÄHLEN
# ============================================================

# Nur volle Jahre 2021–2024 verwenden, man hätte auch mehr daten nehmen können aber da wir so oder so nicht mit totalen datensatzt starten verursachen die paar monate keinen weltuntergang
start = "2021-01-01 00:00:00"
end   = "2024-12-31 23:45:00"   # letzter 15‑Minuten-Slot 2024

df = df.loc[start:end].copy()
print("Zeitraum fürs Modell:", df.index.min(), "→", df.index.max(), "Rows:", len(df))

# ============================================================
# 3) FEATURES & ZIEL
# ============================================================
# nach df = df.loc[start:end].copy()

target_col = "Stromverbrauch"
y = df[target_col].astype(float).copy()

X_all = df.drop(columns=[target_col]).copy()

# Zeitfeatures aus Index ergänzen (falls du sie willst)
X_all["hour"] = X_all.index.hour
X_all["dayofweek"] = X_all.index.dayofweek
X_all["month"] = X_all.index.month
X_all["dayofyear"] = X_all.index.dayofyear

# 1) Zeige, welche object-Spalten es sind
print("Object-Spalten:", X_all.select_dtypes(include="object").columns.tolist())

# 2) Diese object-Spalten aus den Features entfernen
X_all = X_all.drop(columns=X_all.select_dtypes(include="object").columns)

# 3) Jetzt alles in float umwandeln
X = X_all.astype(float)

# 4) NaNs bereinigen
mask = y.notna()
X = X.loc[mask].dropna(axis=0)
y = y.loc[X.index]

print("X shape:", X.shape, " y shape:", y.shape)
print(X.dtypes.value_counts())

# ============================================================
# 4) ZEITLICHER TRAIN/TEST-SPLIT
# ============================================================

train_ratio = 0.8
n_train = int(len(X) * train_ratio)

X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

print("Train:", X_train.index.min(), "→", X_train.index.max(), "(", len(X_train), ")")
print("Test :", X_test.index.min(), "→", X_test.index.max(), "(", len(X_test), ")")

# ============================================================
# 5) MODELL-FUNKTIONEN
# ============================================================
print("Starte RandomForest-Training...")
def random_forest_regressor():
    """Gibt ein vorkonfiguriertes RandomForest-Regressionsmodell zurück."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
print("RandomForest fertig.")
#print("Starte GradientBoosting-Training...")
def gradient_boosting_regressor():
    """Gibt ein vorkonfiguriertes GradientBoosting-Regressionsmodell zurück."""
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


print("GradientBoosting fertig.")
def train_and_evaluate(model, name: str):
    """Trainiert ein Modell, gibt Metriken zurück und druckt sie."""
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("TRAIN:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE : {train_mae:.2f}")
    print(f"  R²  : {train_r2:.3f}")
    print("TEST:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE : {test_mae:.2f}")
    print(f"  R²  : {test_r2:.3f}")

    return {
        "name": name,
        "model": model,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "metrics": {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
        },
    }

# ============================================================
# 6) MODELLE AUSWÄHLEN, TRAINIEREN, VERGLEICHEN
# ============================================================

results = []

# Random Forest aktiv
results.append(train_and_evaluate(random_forest_regressor(), "RandomForest"))

# Gradient Boosting aktiv
results.append(train_and_evaluate(gradient_boosting_regressor(), "GradientBoosting"))

# Wenn du ein Modell NICHT brauchst, diese Zeile einfach in Kommentar setzen:
# results.append(train_and_evaluate(gradient_boosting_regressor(), "GradientBoosting"))

# Übersicht
print("\n=== Modellübersicht (nach Test-RMSE sortiert) ===")
overview = []
for r in results:
    m = r["metrics"]
    overview.append((r["name"], m["test_rmse"], m["test_mae"], m["test_r2"]))

overview_df = pd.DataFrame(overview, columns=["Model", "Test_RMSE", "Test_MAE", "Test_R2"]).sort_values("Test_RMSE")
print(overview_df.to_string(index=False))

# ============================================================
# 7) OPTIONAL: PLOT FÜR BESTES MODELL
# ============================================================

best = overview_df.iloc[0]["Model"]
print("\nBestes Modell laut RMSE:", best)

best_res = [r for r in results if r["name"] == best][0]
df_plot = pd.DataFrame(
    {"y_true": best_res["y_test"], "y_pred": best_res["y_test_pred"]},
    index=best_res["y_test"].index,
)
df_plot = df_plot.last("30D")

plt.figure(figsize=(14, 4))
plt.plot(df_plot.index, df_plot["y_true"], label="Beobachtet", color="blue", linewidth=1)
plt.plot(df_plot.index, df_plot["y_pred"], label="Vorhersage", color="red", linewidth=1, alpha=0.7)
plt.title(f"Stromverbrauch: Beobachtung vs. Vorhersage (letzte 30 Tage) – {best}")
plt.xlabel("Zeit")
plt.ylabel("Stromverbrauch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(DATA_DIR, f"forecast_last_30D_{best}.png")
plt.savefig(plot_path, dpi=100)
plt.close()

print("Plot gespeichert unter:", plot_path)


# ============================================================
# 8) MODELL UND FEATURELISTE SPEICHERN
# ============================================================

model_path = os.path.join(DATA_DIR, "rf_model.joblib")
feat_path = os.path.join(DATA_DIR, "rf_features.npy")

best_model = best_res["model"]

# Modell speichern
dump(best_model, model_path)

# Feature-Namen in gleicher Reihenfolge speichern
np.save(feat_path, X.columns.to_numpy())

print("Modell gespeichert unter:", model_path)
print("Featureliste gespeichert unter:", feat_path)
