import os
import sqlite3

# Fly.io Volume path is /data. If it exists, use it. Otherwise, use local directory.
if os.path.exists('/data'):
    DB_PATH = '/data/market_data.db'
else:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_data.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db_schema():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            symbol TEXT,
            datetime TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, datetime)
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db_schema()
    print(f"Database schema initialized at {DB_PATH}")
