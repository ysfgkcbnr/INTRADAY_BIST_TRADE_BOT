import pandas as pd
from db_utils import get_db_connection
from datetime import timezone, timedelta
TZ_ISTANBUL = timezone(timedelta(hours=3))

conn = get_db_connection()
df = pd.read_sql_query("SELECT symbol, datetime, open, close FROM historical_data", conn)
conn.close()

# Remove 'BIST:' prefix
df['symbol'] = df['symbol'].str.replace('BIST:', '')
# Convert datetime string to datetime object
df['datetime'] = pd.to_datetime(df['datetime'])

# Pivot to get open and close DataFrames
close_df = df.pivot(index='datetime', columns='symbol', values='close')
open_df = df.pivot(index='datetime', columns='symbol', values='open')

# Set timezone
close_df.index = close_df.index.tz_localize(TZ_ISTANBUL)
open_df.index = open_df.index.tz_localize(TZ_ISTANBUL)

# Filter times just like before
time_str = close_df.index.strftime('%H:%M')
valid_mask = ~time_str.isin(['09:30', '18:00'])
close_df = close_df[valid_mask].copy()
open_df = open_df[valid_mask].copy()
print(close_df.tail())
