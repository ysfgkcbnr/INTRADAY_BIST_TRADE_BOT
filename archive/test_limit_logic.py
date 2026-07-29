import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

TZ_ISTANBUL = timezone(timedelta(hours=3))

dates = pd.date_range(start='2022-11-20', end='2022-11-23', freq='D')
times = ['10:00', '17:00', '17:30']

idx = []
for d in dates:
    for t in times:
        if d.weekday() < 5: # Monday is 21st (2022-11-21)
            idx.append(pd.to_datetime(f"{d.strftime('%Y-%m-%d')} {t}").tz_localize(TZ_ISTANBUL))

close_df = pd.DataFrame({'SASA': np.random.rand(len(idx))}, index=idx)
close_df.loc['2022-11-21 17:30', 'SASA'] = 7.38 # Mock 11-21 close

today_str = '2022-11-22'

# Extract just dates
df_dates = pd.Series(close_df.index.date).unique()
df_dates.sort()

prev_day_date = None
for d in reversed(df_dates):
    if str(d) < today_str:
        prev_day_date = d
        break
        
prev_closes = pd.Series(dtype=float)
if prev_day_date is not None:
    prev_day_data = close_df[close_df.index.date == prev_day_date]
    if not prev_day_data.empty:
        prev_closes = prev_day_data.iloc[-1]

print(f"Previous Day Date: {prev_day_date}")
print(f"Prev Closes:\n{prev_closes}")

