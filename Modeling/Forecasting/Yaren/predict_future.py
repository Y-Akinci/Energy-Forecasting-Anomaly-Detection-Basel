from modeling_yaren import pipe, forecast_multistep
import pandas as pd

df = pd.read_csv("processed_merged_features.csv", ...)

df = df.set_index("Start der Messung (UTC)")
df = df.last("7D")   # letzte 7 Tage

forecast = forecast_multistep(pipe, df, steps=96*7)
forecast.to_csv("forecast_next_week.csv")
