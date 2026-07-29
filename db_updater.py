import os
import datetime
from tvDatafeed import TvDatafeed, Interval
from db_utils import get_db_connection
from custom_logger import logger

# Tickers to fetch
TICKERS_BIST = [
    'AEFES', 'AKBNK', 'ASELS', 'BIMAS', 'EKGYO', 'ENKAI', 'EREGL', 'FROTO',
    'GARAN', 'GUBRF', 'ISCTR', 'KCHOL', 'KRDMD', 'MGROS', 'PETKM', 'PGSUS',
    'SAHOL', 'SASA', 'SISE', 'TAVHL', 'TCELL', 'THYAO', 'TOASO', 'TRALT',
    'TTKOM', 'TUPRS', 'VAKBN', 'YKBNK'
]

def update_database_with_latest_bars():
    logger.info(f"[{datetime.datetime.now()}] tvDatafeed üzerinden veri güncellemesi başlatılıyor...")
    try:
        tv = TvDatafeed()
    except Exception as e:
        logger.error(f"tvDatafeed başlatılamadı: {e}")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    
    total_new_rows = 0

    for symbol in TICKERS_BIST:
        try:
            # Fetch last 3 candles of 30-min interval
            df = tv.get_hist(symbol=symbol, exchange='BIST', interval=Interval.in_30_minute, n_bars=3)
            if df is None or df.empty:
                continue
                
            # Iterate through the fetched rows
            for index, row in df.iterrows():
                # index is datetime object from tvDatafeed
                dt_str = index.strftime('%Y-%m-%d %H:%M:%S')
                
                # Check if it already exists to count properly, though INSERT OR IGNORE handles it
                cursor.execute('SELECT 1 FROM historical_data WHERE symbol = ? AND datetime = ?', (f"BIST:{symbol}", dt_str))
                exists = cursor.fetchone()
                
                if not exists:
                    cursor.execute('''
                        INSERT INTO historical_data (symbol, datetime, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (f"BIST:{symbol}", dt_str, row['open'], row['high'], row['low'], row['close'], row.get('volume', 0.0)))
                    total_new_rows += 1
                    
        except Exception as e:
            logger.error(f"{symbol} güncellenirken hata oluştu: {e}")

    conn.commit()
    conn.close()
    logger.info(f"[{datetime.datetime.now()}] Güncelleme tamamlandı. Toplam {total_new_rows} yeni mum veritabanına eklendi.")

if __name__ == '__main__':
    update_database_with_latest_bars()
