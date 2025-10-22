import panda as pd

# Daten einlesen
df_weather_til_2019 = pd.read_csv('"C:\Users\Kerem Akkaya\OneDrive\Desktop\FHNW\3. Semester\Wissensbasierte Systeme und Machine Learning\Wetterdaten_Basel_2010-2019.csv"')
df_weather_from_2020_til_2025 = pd.read_csv('"C:\Users\Kerem Akkaya\OneDrive\Desktop\FHNW\3. Semester\Wissensbasierte Systeme und Machine Learning\Wetterdaten_Basel_2020-2029.csv"')

# Datum und Uhrzeit zu einer Spalte kombinieren
df_weather_til_2019['datetime'] = pd.to_datetime(df_weather_til_2019['date'].astype(str) + ' ' + df_weather['time'].astype(str))
df_weather_from_2020_til_2025['datetime'] = pd.to_datetime(df_weather_from_2020_til_2025['Date'].astype(str) + ' ' + df_power['Time'].astype(str))

# Inner Join auf 'datetime' Spalte 
df_weather_complete = pd.merge(df_weather_til_2019, df_weather_from_2020_til_2025, on='datetime', how='inner')

# Nach Datum und Uhrzeit sortieren (aufsteigend)
df_weather_complete = df_weather_complete.sort_values(by='datetime', ascending=True).reset_index(drop=True)

# Neue Datei erstellen 
df_weather_complete.to_csv('joined_data.csv', index=False)

