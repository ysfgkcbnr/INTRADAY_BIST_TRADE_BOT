import os
import glob
import pandas as pd
from datetime import datetime, timezone, timedelta
from db_utils import get_db_connection

TZ_ISTANBUL = timezone(timedelta(hours=3))

def parse_csvs_to_db():
    csv_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BISTMIXED_*.csv'))
    if not csv_files:
        print("CSV dosyası bulunamadı.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    
    total_rows = 0
    
    for file_path in csv_files:
        # Örnek dosya adı: BISTMIXED_AKBNK, 30.csv
        filename = os.path.basename(file_path)
        try:
            symbol = filename.split('_')[1].split(',')[0].strip()
        except IndexError:
            continue
            
        print(f"İşleniyor: {symbol}...")
        df = pd.read_csv(file_path)
        
        # Sütunlar: time, open, high, low, close
        for index, row in df.iterrows():
            timestamp = row['time']
            # Epoch'u datetime objesine çevir ve formatla
            dt = datetime.fromtimestamp(timestamp, TZ_ISTANBUL)
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO historical_data (symbol, datetime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (f"BIST:{symbol}", dt_str, row['open'], row['high'], row['low'], row['close'], 0.0))
                total_rows += 1
            except Exception as e:
                print(f"Hata ({symbol} - {dt_str}): {e}")
                
    conn.commit()
    conn.close()
    print(f"İşlem tamamlandı! Toplam {total_rows} satır veritabanına aktarıldı.")

if __name__ == "__main__":
    parse_csvs_to_db()
