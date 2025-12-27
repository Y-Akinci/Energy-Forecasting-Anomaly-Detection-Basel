from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    target_col: str = "Stromverbrauch"
    start_ts: str = "2020-08-31 22:00:00"
    end_ts: str = "2024-12-31 23:45:00"
    tz: str = "UTC"

    use_weather: bool = True
    use_lags: bool = True
    use_calendar: bool = True

    test_size: float = 0.30  # 70/30 split

    exclude_weather: tuple[str, ...] = (
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
        'relative Luftfeuchtigkeit_lag15',
    )

    exclude_lags: tuple[str, ...] = (
        "Grundversorgte Kunden_Lag_15min",
        "Freie Kunden_Lag_15min",
        "Lag_15min",
        "Lag_30min",
        "Diff_15min",
    )


def find_project_root(start_dir: Path, marker_dir: str = "data") -> Path:
    root = start_dir.resolve()
    while root != root.parent and not (root / marker_dir).exists():
        root = root.parent
    return root


def load_raw_df(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(
        input_csv,
        sep=";",
        encoding="latin1",
        parse_dates=["Start der Messung (UTC)"],
    )
    df.set_index("Start der Messung (UTC)", inplace=True)
    return df


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def add_calendar_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    # erwartet: Monat, Wochentag, Stunde (Lokal), Tag des Jahres existieren
    df = df.copy()

    df["Monat_sin"] = np.sin(2 * np.pi * df["Monat"] / 12)
    df["Monat_cos"] = np.cos(2 * np.pi * df["Monat"] / 12)

    df["Wochentag_sin"] = np.sin(2 * np.pi * df["Wochentag"] / 7)
    df["Wochentag_cos"] = np.cos(2 * np.pi * df["Wochentag"] / 7)

    df["Stunde_sin"] = np.sin(2 * np.pi * df["Stunde (Lokal)"] / 24)
    df["Stunde_cos"] = np.cos(2 * np.pi * df["Stunde (Lokal)"] / 24)

    df["TagJahr_sin"] = np.sin(2 * np.pi * df["Tag des Jahres"] / 365)
    df["TagJahr_cos"] = np.cos(2 * np.pi * df["Tag des Jahres"] / 365)

    return df


def build_feature_list(df: pd.DataFrame, cfg: DataConfig) -> tuple[list[str], list[str], list[str]]:
    calendar_features = [
        "Monat_sin", "Monat_cos",
        "Wochentag_sin", "Wochentag_cos",
        "Stunde_sin", "Stunde_cos",
        "TagJahr_sin", "TagJahr_cos",
        "Woche des Jahres", "IstArbeitstag", "IstSonntag",
    ]

    lag_features_all = [
        "Lag_15min", "Lag_30min", "Lag_1h", "Lag_24h",
        "Grundversorgte Kunden_Lag_15min",
        "Freie Kunden_Lag_15min",
        "Diff_15min",
    ]
    lag_features = [c for c in lag_features_all if c not in cfg.exclude_lags]

    weather_all = [c for c in df.columns if c.endswith("_lag15") and c not in lag_features_all]
    weather_features = [c for c in weather_all if c not in cfg.exclude_weather]

    features: list[str] = []
    if cfg.use_calendar:
        features += calendar_features
    if cfg.use_lags:
        features += lag_features
    if cfg.use_weather:
        features += weather_features

    features = [f for f in features if f in df.columns]
    numeric_features = features[:]  # alles numerisch

    return features, numeric_features, weather_features


def prepare_df(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = ensure_utc_index(df)

    start_ts = pd.Timestamp(cfg.start_ts, tz=cfg.tz)
    end_ts = pd.Timestamp(cfg.end_ts, tz=cfg.tz)
    df = df.loc[start_ts:end_ts].copy()

    df = add_calendar_sin_cos(df)

    features, numeric_features, weather_features = build_feature_list(df, cfg)

    # Drop nur Anzeige-/Originalspalten, nicht Features
    drop_cols = [
        "Datum (Lokal)", "Zeit (Lokal)",
        "Monat", "Wochentag", "Stunde (Lokal)",
        "Tag des Jahres", "Quartal",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, numeric_features, weather_features


def time_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def load_dataset_and_split(
    *,
    cfg: DataConfig | None = None,
    input_csv: Path | None = None,
    verbose: bool = True,
) -> dict:
    cfg = cfg or DataConfig()

    if input_csv is None:
        base_dir = Path(__file__).resolve().parent
        root = find_project_root(base_dir, marker_dir="data")
        input_csv = root / "data" / "processed_merged_features.csv"
    else:
        root = input_csv.resolve().parents[0]

    if verbose:
        print("ROOT:", root)
        print("INPUT_CSV:", input_csv)
        print("Exists:", input_csv.exists())

    df_raw = load_raw_df(input_csv)
    df_full, numeric_features, weather_features = prepare_df(df_raw, cfg)
    train_df, test_df = time_split(df_full, cfg.test_size)

    if verbose:
        print("Index-TZ:", df_full.index.tz)
        print("Zeitraum:", df_full.index.min(), "→", df_full.index.max())
        print("Train:", train_df.index.min(), "→", train_df.index.max(), "|", len(train_df))
        print("Test :", test_df.index.min(), "→", test_df.index.max(), "|", len(test_df))
        print("Aktive Features:", len(numeric_features))

    return {
        "root": root,
        "df_full": df_full,
        "train_df": train_df,
        "test_df": test_df,
        "numeric_features": numeric_features,
        "weather_features": weather_features,
        "cfg": cfg,
    }
