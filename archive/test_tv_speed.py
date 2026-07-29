import time
from tvDatafeed import TvDatafeed, Interval
tv = TvDatafeed()
tickers = ['AEFES', 'AKBNK', 'ASELS', 'BIMAS', 'EKGYO', 'ENKAI', 'EREGL', 'FROTO',
    'GARAN', 'GUBRF', 'ISCTR', 'KCHOL', 'KRDMD', 'MGROS', 'PETKM', 'PGSUS',
    'SAHOL', 'SASA', 'SISE', 'TAVHL', 'TCELL', 'THYAO', 'TOASO', 'TRALT',
    'TTKOM', 'TUPRS', 'VAKBN', 'YKBNK']
start = time.time()
for t in tickers:
    tv.get_hist(symbol=t, exchange='BIST', interval=Interval.in_30_minute, n_bars=300)
print(f"Time taken: {time.time() - start:.2f} seconds")
