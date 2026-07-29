import pandas as pd
import numpy as np

# Mock data
df = pd.DataFrame({
    'ticker': ['A']*32,
    'date': ['2022-01-01']*16 + ['2022-01-02']*16,
    'interval_idx': list(range(1, 17)) * 2,
    'close': np.random.rand(32),
    'open': np.random.rand(32)
})

# Get last close of each day
last_closes = df[df['interval_idx'] == 16][['ticker', 'date', 'close']].copy()
last_closes['prev_day_close'] = last_closes.groupby('ticker')['close'].shift(1)
last_closes = last_closes.drop('close', axis=1)

df = df.merge(last_closes, on=['ticker', 'date'], how='left')
df['change_at_open'] = (df['open'] / df['prev_day_close']) - 1.0

print(df[df['interval_idx'].isin([1, 16])])
