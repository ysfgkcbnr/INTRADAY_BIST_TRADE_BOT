#!/usr/bin/env python3
"""
STRATEJİ 1: Test A1 (1x1) — Sadece 17:30 Barı (30 Dk İşlem)
================================================================================
Strateji Özeti:
- Her gün 17:00'da webhook ile tetiklenir, pozisyon açar.
- Her gün 17:30'da webhook ile tetiklenir, pozisyon kapatır.
- Sinyal: Geçmiş K=10 günün 17:00-17:30 getirileri üzerinden 1/k Z-Skor.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from db_utils import get_db_connection


warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE_DIR, 'portfolio_a1_1x1.json')
TZ_ISTANBUL = timezone(timedelta(hours=3))

INITIAL_EQUITY = 100000.0
COST_BPS = 0.00020
K_DAYS = 10

TICKERS_BIST = [
    'AEFES', 'AKBNK', 'ASELS', 'BIMAS', 'EKGYO', 'ENKAI', 'EREGL', 'FROTO',
    'GARAN', 'GUBRF', 'ISCTR', 'KCHOL', 'KRDMD', 'MGROS', 'PETKM', 'PGSUS',
    'SAHOL', 'SASA', 'SISE', 'TAVHL', 'TCELL', 'THYAO', 'TOASO', 'TRALT',
    'TTKOM', 'TUPRS', 'VAKBN', 'YKBNK'
]

YF_MAP = {t: f"{t}.IS" for t in TICKERS_BIST}
REV_YF_MAP = {v: k for k, v in YF_MAP.items()}


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    portfolio = {
        'strategy_name': 'Test A1 (1x1) - 30 Dk Vur-Kaç',
        'last_updated': datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S'),
        'initial_capital': INITIAL_EQUITY,
        'cash': INITIAL_EQUITY,
        'equity': INITIAL_EQUITY,
        'total_realized_pnl': 0.0,
        'total_unrealized_pnl': 0.0,
        'total_commission_paid': 0.0,
        'active_positions': {},
        'trade_history': [],
        'equity_history': []
    }
    save_portfolio(portfolio)
    return portfolio


def save_portfolio(portfolio):
    portfolio['last_updated'] = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def reset_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        os.remove(PORTFOLIO_FILE)
    load_portfolio()
    print("Strateji 1 (Test A1 1x1) Sanal Portföyü 100.000 TL'ye sıfırlandı.")


def fetch_data():
    try:
        conn = get_db_connection()
        # Sadece son 14 günün verisini almak için basit bir tarih filtresi eklenebilir, 
        # ancak pandas ile filtrelemek de hızlıdır. Tüm veriyi çekip son günleri alıyoruz.
        df = pd.read_sql_query("SELECT symbol, datetime, open, close FROM historical_data", conn)
        conn.close()

        if df.empty:
            return None

        # 'BIST:' önekini temizle
        df['symbol'] = df['symbol'].str.replace('BIST:', '')
        # datetime string'lerini datetime objesine çevir
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Pivot tablolara çevir (satırlar: zaman, sütunlar: hisse kodu)
        close_df = df.pivot(index='datetime', columns='symbol', values='close')
        open_df = df.pivot(index='datetime', columns='symbol', values='open')

        # Zaman dilimini Istanbul'a ayarla
        close_df.index = close_df.index.tz_localize(TZ_ISTANBUL)
        open_df.index = open_df.index.tz_localize(TZ_ISTANBUL)

        # 14 günlük filtre
        fourteen_days_ago = datetime.now(TZ_ISTANBUL) - timedelta(days=14)
        close_df = close_df[close_df.index >= fourteen_days_ago]
        open_df = open_df[open_df.index >= fourteen_days_ago]

        # 09:30 ve 18:00 (kapanış) mumlarını filtrele (eski yfinance kalıntısı)
        time_str = close_df.index.strftime('%H:%M')
        valid_mask = ~time_str.isin(['09:30', '18:00'])
        close_df = close_df[valid_mask].copy()
        open_df = open_df[valid_mask].copy()

        r_oc_df = np.log(close_df / open_df)
        return {'close': close_df, 'open': open_df, 'r_oc': r_oc_df}
    except Exception as e:
        print(f"Veri çekme hatası (Veritabanı): {e}")
        return None


def run_entry():
    now_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{now_str}] Strateji 1 (A1) ENTRY (17:00) Başladı...")

    portfolio = load_portfolio()
    data = fetch_data()
    if data is None:
        return

    r_oc_df = data['r_oc']
    close_1730 = r_oc_df[r_oc_df.index.strftime('%H:%M') == '17:00']

    if len(close_1730) < K_DAYS:
        print("Yeterli geçmiş veri yok.")
        return

    # Sinyal SADECE geçmiş K güne bakılarak hesaplanır (Bugün hariç)
    # Eğer close_1730'un son satırı bugüne aitse (17:00 Yahoo'da oluştuysa), onu alma!
    today_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d')
    if close_1730.index[-1].strftime('%Y-%m-%d') == today_str:
        recent_k = close_1730.iloc[-(K_DAYS+1):-1]
    else:
        recent_k = close_1730.iloc[-K_DAYS:]

    hist_zscores = []
    weights = []

    for k in range(1, K_DAYS + 1):
        r_k = recent_k.iloc[-k].values
        valid_m = np.isfinite(r_k)
        if valid_m.sum() > 5:
            z_k = (r_k - np.nanmean(r_k)) / (np.nanstd(r_k) + 1e-8)
            hist_zscores.append(z_k)
            weights.append(1.0 / k)

    if not hist_zscores:
        print("Geçmiş z-skor hesaplanamadı.")
        return

    w_arr = np.array(weights) / np.sum(weights)
    sig_series = pd.Series(np.tensordot(w_arr, np.array(hist_zscores), axes=(0, 0)), index=r_oc_df.columns)

    # 17:00 Webhook Fiyatlarını Oku (open_prices.json)
    open_prices = {}
    if os.path.exists(os.path.join(BASE_DIR, 'open_prices.json')):
        with open(os.path.join(BASE_DIR, 'open_prices.json'), 'r') as f:
            open_prices = json.load(f)
    
    if not open_prices:
        print("open_prices.json bulunamadı. Webhook gelmedi mi?")
        return

    valid_stocks = sig_series.dropna().index.intersection(open_prices.keys())
    sorted_stocks = sig_series[valid_stocks].sort_values(ascending=False)

    top_1 = sorted_stocks.index[0]
    bottom_1 = sorted_stocks.index[-1]

    price_top = float(open_prices[top_1])
    price_bottom = float(open_prices[bottom_1])

    msg_top = f"Top 1 Long Adayı  : {top_1} (Sinyal: {sorted_stocks[top_1]:+.3f}, Giriş: {price_top:.2f} TL)"
    msg_bot = f"Bottom 1 Short Adayı: {bottom_1} (Sinyal: {sorted_stocks[bottom_1]:+.3f}, Giriş: {price_bottom:.2f} TL)"
    print(msg_top)
    print(msg_bot)

    signal_log_path = os.path.join(BASE_DIR, 'signals_a1_1x1.log')
    with open(signal_log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{now_str}] ENTRY\n{msg_top}\n{msg_bot}\n\n")

    total_equity = portfolio['equity']
    target_shares_long = int(total_equity / price_top) if price_top > 0 else 0
    target_shares_short = int(total_equity / price_bottom) if price_bottom > 0 else 0

    # Mevcut pozisyon varsa temizle (olmamasi lazim)
    portfolio['active_positions'] = {}

    if target_shares_long > 0:
        cost = (price_top * target_shares_long) * COST_BPS
        portfolio['cash'] -= cost
        portfolio['total_commission_paid'] += cost
        portfolio['active_positions'][top_1] = {
            'side': 'LONG', 'shares': target_shares_long, 'entry_price': price_top, 'curr_price': price_top, 'unrealized_pnl': 0.0
        }

    if target_shares_short > 0:
        cost = (price_bottom * target_shares_short) * COST_BPS
        portfolio['cash'] -= cost
        portfolio['total_commission_paid'] += cost
        portfolio['active_positions'][bottom_1] = {
            'side': 'SHORT', 'shares': target_shares_short, 'entry_price': price_bottom, 'curr_price': price_bottom, 'unrealized_pnl': 0.0
        }

    save_portfolio(portfolio)
    print("A1 Entry Tamamlandı.")


def run_exit():
    now_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{now_str}] Strateji 1 (A1) EXIT (17:30) Başladı...")

    portfolio = load_portfolio()
    
    close_prices = {}
    if os.path.exists(os.path.join(BASE_DIR, 'close_prices.json')):
        with open(os.path.join(BASE_DIR, 'close_prices.json'), 'r') as f:
            close_prices = json.load(f)
            
    if not close_prices:
        print("close_prices.json bulunamadı. Webhook gelmedi mi?")
        return

    # Tüm pozisyonları kapat
    for t in list(portfolio['active_positions'].keys()):
        if t in close_prices:
            pos = portfolio['active_positions'].pop(t)
            exit_price = float(close_prices[t])
            realized = (exit_price - pos['entry_price']) * pos['shares'] if pos['side'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['shares']
            cost = (exit_price * pos['shares']) * COST_BPS

            portfolio['cash'] += realized - cost
            portfolio['total_realized_pnl'] += realized
            portfolio['total_commission_paid'] += cost

            if 'trade_history' not in portfolio:
                portfolio['trade_history'] = []
            
            portfolio['trade_history'].append({
                'ticker': t,
                'side': pos['side'],
                'shares': pos['shares'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'realized_pnl': realized,
                'commission_paid': cost,
                'close_time': now_str
            })
            print(f"{t} Exit: {exit_price:.2f} TL | PnL: {realized:.2f} TL")

    portfolio['equity'] = portfolio['cash']
    
    if 'equity_history' not in portfolio:
        portfolio['equity_history'] = []
    portfolio['equity_history'].append({
        'time': now_str,
        'equity': portfolio['equity']
    })

    save_portfolio(portfolio)
    print(f"A1 Exit Tamamlandı. Güncel Sermaye: {portfolio['equity']:.2f}")


def report():
    p = load_portfolio()
    print("=" * 70)
    print("STRATEJİ 1: Test A1 (1x1) PORTFÖY RAPORU")
    print("=" * 70)
    print(f"Güncel Sanal Özkaynak: {p['equity']:,.2f} TL")
    print(f"Aktif Pozisyonlar   : {list(p['active_positions'].keys())}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Strateji 1 A1 Botu')
    parser.add_argument('--run_entry', action='store_true')
    parser.add_argument('--run_exit', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--reset', action='store_true')

    args = parser.parse_args()
    if args.reset:
        reset_portfolio()
    elif args.report:
        report()
    elif args.run_entry:
        run_entry()
        report()
    elif args.run_exit:
        run_exit()
        report()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
