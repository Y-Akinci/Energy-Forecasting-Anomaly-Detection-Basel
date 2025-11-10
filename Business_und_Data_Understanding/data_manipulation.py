import pandas as pd

#Daten einlesen
df_weather_til_2019 = pd.read_csv(r'C:\Users\Kerem Akkaya\OneDrive\Desktop\FHNW\3. Semester\Wissensbasierte Systeme und Machine Learning\Wetterdaten_Basel_2010-2019.csv', encoding='latin1')
df_weather_from_2020_til_2025 = pd.read_csv(r'C:\Users\Kerem Akkaya\OneDrive\Desktop\FHNW\3. Semester\Wissensbasierte Systeme und Machine Learning\Wetterdaten_Basel_2020-2029.csv', encoding='latin1')

#Beide DataFrames zusammenführen
df_weather_complete = pd.concat([df_weather_til_2019, df_weather_from_2020_til_2025], ignore_index=True)

#Nach Datum und Uhrzeit sortieren
df_weather_complete = df_weather_complete.sort_values(by=['Date', 'Time'], ascending=True).reset_index(drop=True)

#Neue Datei erstellen
df_weather_complete.to_csv('complete_weather_data.csv', index=False)

