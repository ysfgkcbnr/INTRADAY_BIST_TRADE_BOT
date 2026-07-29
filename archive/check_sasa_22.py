import pandas as pd
import numpy as np

df = pd.read_csv('BISTMIXED_SASA, 30.csv')
df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/Istanbul')
df['date'] = df['datetime'].dt.date
df['time_str'] = df['datetime'].dt.strftime('%H:%M')

sasa_nov = df[(df['date'].astype(str) >= '2022-11-21') & (df['date'].astype(str) <= '2022-11-23')]
print(sasa_nov[['date', 'time_str', 'open', 'high', 'low', 'close']])
