import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETUP
# ============================================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Gehe ein Verzeichnis hoch zum Repo-Root
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "Data")

INPUT_FILE = os.path.join(DATA_DIR, "processed_merged_features.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "exploration_plots")

# Output-Verzeichnis erstellen
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("🔍 DATA EXPLORATION - Energy Forecasting Basel")
print("=" * 70)

# ============================================================
# 1) DATEN LADEN
# ============================================================

print("\n[1/8] Lade Daten...")
df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    encoding="latin1",
    parse_dates=["DateTime"],  
    index_col="DateTime"        
)
df = df.sort_index()

print(f"✓ Geladen: {len(df)} Zeilen, Zeitraum: {df.index.min()} bis {df.index.max()}")

# Target-Variable
target = "Stromverbrauch"

# ============================================================
# 2) ZEITREIHEN-PLOTS (Muster über Zeit)
# ============================================================

print("\n[2/8] Erstelle Zeitreihen-Plots...")

# 2.1 Gesamter Zeitraum
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df.index, df[target], linewidth=0.5, alpha=0.8, color='steelblue')
ax.set_title('Stromverbrauch über gesamten Zeitraum', fontsize=14, fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_zeitreihe_gesamt.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Gesamtzeitreihe gespeichert")

# 2.2 Zoom: 1 Monat (Januar 2024)
fig, ax = plt.subplots(figsize=(16, 6))
df_month = df.loc['2024-01-01':'2024-01-31']
ax.plot(df_month.index, df_month[target], linewidth=1, color='steelblue', marker='o', markersize=2)
ax.set_title('Stromverbrauch - Januar 2024 (Wochenmuster sichtbar)', fontsize=14, fontweight='bold')
ax.set_xlabel('Datum')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_zeitreihe_1monat.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1-Monats-Zoom gespeichert")

# 2.3 Zoom: 1 Woche (erste Woche Januar 2024)
fig, ax = plt.subplots(figsize=(16, 6))
df_week = df.loc['2024-01-01':'2024-01-07']
ax.plot(df_week.index, df_week[target], linewidth=2, color='steelblue', marker='o', markersize=4)
ax.set_title('Stromverbrauch - Erste Woche Januar 2024 (Tag/Nacht-Zyklus)', fontsize=14, fontweight='bold')
ax.set_xlabel('Datum und Uhrzeit')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_zeitreihe_1woche.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1-Wochen-Zoom gespeichert")

# 2.4 Vergleich Sommer vs Winter
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Winter (Januar 2024)
winter = df.loc['2024-01-01':'2024-01-31']
ax1.plot(winter.index, winter[target], linewidth=1, color='navy', alpha=0.8)
ax1.set_title('Winter: Januar 2024', fontsize=12, fontweight='bold')
ax1.set_ylabel('Stromverbrauch (kWh)')
ax1.grid(alpha=0.3)

# Sommer (Juli 2024)
summer = df.loc['2024-07-01':'2024-07-31']
ax2.plot(summer.index, summer[target], linewidth=1, color='orangered', alpha=0.8)
ax2.set_title('Sommer: Juli 2024', fontsize=12, fontweight='bold')
ax2.set_xlabel('Datum')
ax2.set_ylabel('Stromverbrauch (kWh)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_vergleich_winter_sommer.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Winter/Sommer-Vergleich gespeichert")

# ============================================================
# 3) PERIODISCHE MUSTER
# ============================================================

print("\n[3/8] Analysiere periodische Muster...")

# Lokale Zeit-Features aus UTC-Index extrahieren
df_local = df.copy()
df_local['hour'] = df_local.index.hour
df_local['dayofweek'] = df_local.index.dayofweek  # 0=Montag, 6=Sonntag
df_local['month'] = df_local.index.month

# 3.1 Heatmap: Stunde x Wochentag
pivot_hour_day = df_local.pivot_table(
    values=target, 
    index='hour', 
    columns='dayofweek', 
    aggfunc='mean'
)
pivot_hour_day.columns = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pivot_hour_day, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Ø Verbrauch (kWh)'})
ax.set_title('Durchschnittlicher Stromverbrauch: Stunde x Wochentag', fontsize=14, fontweight='bold')
ax.set_xlabel('Wochentag')
ax.set_ylabel('Stunde des Tages')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_heatmap_stunde_wochentag.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Heatmap Stunde x Wochentag gespeichert")

# 3.2 Boxplot: Verbrauch pro Wochentag
fig, ax = plt.subplots(figsize=(14, 6))
df_local['dayname'] = df_local.index.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.boxplot(data=df_local, x='dayname', y=target, order=day_order, ax=ax, palette='Set2')
ax.set_title('Stromverbrauch pro Wochentag', fontsize=14, fontweight='bold')
ax.set_xlabel('Wochentag')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.grid(alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_boxplot_wochentag.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Boxplot Wochentag gespeichert")

# 3.3 Boxplot: Verbrauch pro Monat
fig, ax = plt.subplots(figsize=(14, 6))
month_names = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
df_local['month_name'] = df_local['month'].apply(lambda x: month_names[x-1])
sns.boxplot(data=df_local, x='month_name', y=target, order=month_names, ax=ax, palette='coolwarm')
ax.set_title('Stromverbrauch pro Monat (Saisonalität)', fontsize=14, fontweight='bold')
ax.set_xlabel('Monat')
ax.set_ylabel('Stromverbrauch (kWh)')
ax.grid(alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_boxplot_monat.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Boxplot Monat gespeichert")

# 3.4 Line-Plot: Durchschnittlicher Verbrauch pro Stunde
fig, ax = plt.subplots(figsize=(14, 6))
hourly_avg = df_local.groupby('hour')[target].mean()
ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8, color='steelblue')
ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='steelblue')
ax.set_title('Durchschnittliches Tagesprofil (alle Tage)', fontsize=14, fontweight='bold')
ax.set_xlabel('Stunde des Tages')
ax.set_ylabel('Ø Stromverbrauch (kWh)')
ax.set_xticks(range(0, 24))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_tagesprofil.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Tagesprofil gespeichert")

# ============================================================
# 4) FEATURE-EINFLUSS AUF STROMVERBRAUCH
# ============================================================

print("\n[4/8] Analysiere Feature-Einfluss...")

# Relevante Wetterfeatures identifizieren
weather_features = [col for col in df.columns if '_lag15' in col]

# 4.1 Temperatur vs. Verbrauch
temp_col = [col for col in weather_features if 'Lufttemperatur 2 m' in col]
if temp_col:
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df[temp_col[0]], 
        df[target], 
        c=df_local['month'], 
        cmap='coolwarm', 
        alpha=0.3, 
        s=1
    )
    ax.set_title('Temperatur vs. Stromverbrauch (gefärbt nach Monat)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lufttemperatur 2m [°C] (lag 15min)')
    ax.set_ylabel('Stromverbrauch (kWh)')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Monat')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_scatter_temperatur.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Temperatur-Scatter gespeichert")

# 4.2 Globalstrahlung vs. Verbrauch
global_col = [col for col in weather_features if 'Globalstrahlung' in col]
if global_col:
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df[global_col[0]], 
        df[target], 
        c=df_local['hour'], 
        cmap='viridis', 
        alpha=0.3, 
        s=1
    )
    ax.set_title('Globalstrahlung vs. Stromverbrauch (gefärbt nach Stunde)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Globalstrahlung [W/m²] (lag 15min)')
    ax.set_ylabel('Stromverbrauch (kWh)')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Stunde')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '10_scatter_globalstrahlung.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Globalstrahlung-Scatter gespeichert")

# 4.3 Windgeschwindigkeit vs. Verbrauch
wind_col = [col for col in weather_features if 'Windgeschwindigkeit skalar' in col]
if wind_col:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df[wind_col[0]], df[target], alpha=0.3, s=1, color='steelblue')
    ax.set_title('Windgeschwindigkeit vs. Stromverbrauch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Windgeschwindigkeit [m/s] (lag 15min)')
    ax.set_ylabel('Stromverbrauch (kWh)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '11_scatter_wind.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Wind-Scatter gespeichert")

# 4.4 Korrelations-Heatmap (Top Features)
# Wähle die wichtigsten numerischen Features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Entferne kategorische und redundante Features
exclude_cols = ['Monat', 'Wochentag', 'Tag des Jahres', 'Quartal', 'Woche des Jahres']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Berechne Korrelation mit Target
correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
top_features = correlations.head(20).index.tolist()

fig, ax = plt.subplots(figsize=(14, 12))
corr_matrix = df[top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, 
    mask=mask, 
    annot=True, 
    fmt='.2f', 
    cmap='RdBu_r', 
    center=0, 
    square=True,
    ax=ax,
    cbar_kws={'label': 'Korrelation'}
)
ax.set_title('Korrelations-Heatmap (Top 20 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_korrelation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Korrelations-Heatmap gespeichert")

# ============================================================
# 5) VERTEILUNGEN & OUTLIERS
# ============================================================

print("\n[5/8] Analysiere Verteilungen...")

# 5.1 Histogram: Stromverbrauch
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df[target].dropna(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title('Verteilung des Stromverbrauchs', fontsize=14, fontweight='bold')
ax.set_xlabel('Stromverbrauch (kWh)')
ax.set_ylabel('Häufigkeit')
ax.axvline(df[target].mean(), color='red', linestyle='--', linewidth=2, label=f'Mittelwert: {df[target].mean():.0f}')
ax.axvline(df[target].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[target].median():.0f}')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '13_histogram_verbrauch.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Histogram gespeichert")

# 5.2 Boxplot mit Outliers
fig, ax = plt.subplots(figsize=(14, 6))
ax.boxplot(df[target].dropna(), vert=False, widths=0.5)
ax.set_title('Stromverbrauch - Boxplot mit Outliers', fontsize=14, fontweight='bold')
ax.set_xlabel('Stromverbrauch (kWh)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '14_boxplot_outliers.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Boxplot Outliers gespeichert")

# 5.3 Q-Q Plot
fig, ax = plt.subplots(figsize=(10, 10))
stats.probplot(df[target].dropna(), dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Normalverteilung?', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '15_qq_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Q-Q Plot gespeichert")

# ============================================================
# 6) SAISONALE DEKOMPOSITION
# ============================================================

print("\n[6/8] Führe saisonale Dekomposition durch...")

# Resampling auf stündliche Daten für schnellere Berechnung
df_hourly = df[target].resample('1H').mean().dropna()

# Nehme 1 Jahr für Dekomposition (2023)
df_decomp = df_hourly.loc['2023-01-01':'2023-12-31']

if len(df_decomp) > 365 * 24 * 0.9:  # Mindestens 90% der Daten vorhanden
    try:
        decomposition = seasonal_decompose(
            df_decomp, 
            model='additive', 
            period=24*7  # Wöchentliche Saisonalität
        )
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # Original
        axes[0].plot(decomposition.observed, linewidth=0.8, color='black')
        axes[0].set_title('Original-Zeitreihe (2023)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Verbrauch')
        axes[0].grid(alpha=0.3)
        
        # Trend
        axes[1].plot(decomposition.trend, linewidth=1.5, color='red')
        axes[1].set_title('Trend', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Trend')
        axes[1].grid(alpha=0.3)
        
        # Saisonalität
        axes[2].plot(decomposition.seasonal, linewidth=0.8, color='green')
        axes[2].set_title('Saisonalität (Wöchentlich)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Saisonalität')
        axes[2].grid(alpha=0.3)
        
        # Residuen
        axes[3].plot(decomposition.resid, linewidth=0.5, color='blue', alpha=0.7)
        axes[3].set_title('Residuen', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Residuen')
        axes[3].set_xlabel('Zeit')
        axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '16_saisonale_dekomposition.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saisonale Dekomposition gespeichert")
    except Exception as e:
        print(f"  ⚠ Saisonale Dekomposition fehlgeschlagen: {e}")
else:
    print("  ⚠ Nicht genug Daten für saisonale Dekomposition")

# ============================================================
# 7) LAG-FEATURE ANALYSE
# ============================================================

print("\n[7/8] Analysiere Lag-Features...")

# 7.1 Lag_15min vs. aktueller Verbrauch
if 'Lag_15min' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df['Lag_15min'], df[target], alpha=0.2, s=1, color='steelblue')
    
    # Regressionslinie
    z = np.polyfit(df['Lag_15min'].dropna(), df[target].loc[df['Lag_15min'].dropna().index], 1)
    p = np.poly1d(z)
    ax.plot(df['Lag_15min'].dropna().sort_values(), 
            p(df['Lag_15min'].dropna().sort_values()), 
            "r--", linewidth=2, label=f'Regression: y={z[0]:.2f}x+{z[1]:.0f}')
    
    ax.set_title('Autokorrelation: Lag 15min vs. aktueller Verbrauch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Verbrauch vor 15 min (kWh)')
    ax.set_ylabel('Aktueller Verbrauch (kWh)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '17_lag_15min_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Lag-Scatter gespeichert")

# 7.2 Autocorrelation Plot (ACF)
fig, ax = plt.subplots(figsize=(14, 6))
# Nehme nur jede 4. Beobachtung (stündlich) für schnellere Berechnung
plot_acf(df[target].dropna()[::4], lags=7*24, ax=ax, alpha=0.05)
ax.set_title('Autokorrelationsfunktion (ACF) - Stündliche Daten, 7 Tage', fontsize=14, fontweight='bold')
ax.set_xlabel('Lag (Stunden)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '18_acf_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ACF Plot gespeichert")

# ============================================================
# 8) WETTERFEATURES ÜBER ZEIT
# ============================================================

print("\n[8/8] Erstelle Wetter-Overlay-Plots...")

# 8.1 Temperatur + Verbrauch (2023)
temp_col = [col for col in weather_features if 'Lufttemperatur 2 m' in col]
if temp_col:
    df_2023 = df.loc['2023-01-01':'2023-12-31']
    
    fig, ax1 = plt.subplots(figsize=(16, 6))
    
    color1 = 'steelblue'
    ax1.set_xlabel('Zeit')
    ax1.set_ylabel('Stromverbrauch (kWh)', color=color1)
    ax1.plot(df_2023.index, df_2023[target], color=color1, linewidth=0.5, alpha=0.7, label='Stromverbrauch')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'orangered'
    ax2.set_ylabel('Temperatur (°C)', color=color2)
    ax2.plot(df_2023.index, df_2023[temp_col[0]], color=color2, linewidth=0.8, alpha=0.8, label='Temperatur')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Stromverbrauch und Temperatur über das Jahr 2023', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '19_verbrauch_temperatur_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Temperatur-Overlay gespeichert")

# 8.2 Globalstrahlung + Verbrauch (Sommer 2023)
global_col = [col for col in weather_features if 'Globalstrahlung' in col]
if global_col:
    df_summer = df.loc['2023-07-01':'2023-07-31']
    
    fig, ax1 = plt.subplots(figsize=(16, 6))
    
    color1 = 'steelblue'
    ax1.set_xlabel('Zeit')
    ax1.set_ylabel('Stromverbrauch (kWh)', color=color1)
    ax1.plot(df_summer.index, df_summer[target], color=color1, linewidth=1, alpha=0.8, label='Stromverbrauch')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'gold'
    ax2.set_ylabel('Globalstrahlung (W/m²)', color=color2)
    ax2.plot(df_summer.index, df_summer[global_col[0]], color=color2, linewidth=1, alpha=0.8, label='Globalstrahlung')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Stromverbrauch und Globalstrahlung - Juli 2023', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '20_verbrauch_solar_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Globalstrahlung-Overlay gespeichert")

# ============================================================
# FERTIG!
# ============================================================

print("\n" + "=" * 70)
print("✅ ALLE PLOTS ERFOLGREICH ERSTELLT!")
print(f"📁 Gespeichert in: {OUTPUT_DIR}")
print("=" * 70)
print("\nÜbersicht der erstellten Plots:")
print("  01 - Zeitreihe gesamt")
print("  02 - Zeitreihe 1 Monat")
print("  03 - Zeitreihe 1 Woche")
print("  04 - Vergleich Winter/Sommer")
print("  05 - Heatmap Stunde x Wochentag ⭐")
print("  06 - Boxplot Wochentag")
print("  07 - Boxplot Monat")
print("  08 - Tagesprofil")
print("  09 - Scatter Temperatur ⭐")
print("  10 - Scatter Globalstrahlung")
print("  11 - Scatter Wind")
print("  12 - Korrelations-Heatmap ⭐")
print("  13 - Histogram Verbrauch")
print("  14 - Boxplot Outliers")
print("  15 - Q-Q Plot")
print("  16 - Saisonale Dekomposition ⭐")
print("  17 - Lag-Scatter")
print("  18 - ACF Plot")
print("  19 - Temperatur-Overlay ⭐")
print("  20 - Globalstrahlung-Overlay")
print("\n⭐ = Besonders wichtig für Präsentation")