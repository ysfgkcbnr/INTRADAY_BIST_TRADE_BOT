#!/usr/bin/env python3
"""
STRATEJİ 4: Test C1 (1x1) — 1/10 Overlapping 10-Gün Taşıma (17:30 Rebalance)
================================================================================
Strateji Özeti:
- 17:30 periyodiklik sinyali ile açılan 1x1 (Top 1 Long / Bottom 1 Short) pozisyonlarını
  1/10 çakışmalı (overlapping) yapıda 10 gün boyunca kesintisiz taşır.
- Rebalance her akşam tam 17:30 kapanışından önceki 17:30 fiyatı ile (open_prices.json) yapılır.
- Sinyal: Geçmiş K=10 günün 17:30 periyodundaki getirileri (1/k z-score).
- Seçim: En yüksek 1 hisse LONG (+100k TL Notional), en düşük 1 hisse SHORT (-100k TL Notional).
- Sanal Para: 100.000 TL Özkaynak (2.0x VİOP Kaldıraçlı).
- Komisyon: 10 binde 2 (%0.020 / 2.0 BPS).

Kullanım:
  python3 strateji_4_test_c1_1x1.py --run       # Tek bir 30 dk canlı güncelleme ve rebalance
  python3 strateji_4_test_c1_1x1.py --report    # Sanal portföy durum raporu
  python3 strateji_4_test_c1_1x1.py --reset     # Sanal bakiye sıfırlama (100.000 TL)
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
PORTFOLIO_FILE = os.path.join(BASE_DIR, 'portfolio_c1_1x1.json')
TZ_ISTANBUL = timezone(timedelta(hours=3))

INITIAL_EQUITY = 100000.0
COST_BPS = 0.00035
K_DAYS = 10
N_SIZE = 1

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
        'strategy_name': 'Test C1 (1x1) - 17:30 Rebalance',
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
    print("Strateji 4 (Test C1 1x1 17:30 Rebalance) Sanal Portföyü 100.000 TL'ye sıfırlandı.")


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

        # 30 günlük filtre
        thirty_days_ago = datetime.now(TZ_ISTANBUL) - timedelta(days=30)
        close_df = close_df[close_df.index >= thirty_days_ago]
        open_df = open_df[open_df.index >= thirty_days_ago]

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


def run_strategy():
    now_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{now_str}] Strateji 4 (Test C1 1x1 17:30 Rebalance) Canlı Sinyal & Rebalance...")

    portfolio = load_portfolio()
    data = fetch_data()
    if data is None:
        return

    r_oc_df = data['r_oc']
    close_df = data['close']
    close_1730 = r_oc_df[r_oc_df.index.strftime('%H:%M') == '17:30']

    if len(close_1730) < K_DAYS:
        print("Yeterli geçmiş veri yok.")
        return

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

    w_arr = np.array(weights) / np.sum(weights)
    sig_series = pd.Series(np.tensordot(w_arr, np.array(hist_zscores), axes=(0, 0)), index=r_oc_df.columns)

    # 17:30 Webhook Fiyatlarını Oku (open_prices.json)
    open_prices = {}
    if os.path.exists(os.path.join(BASE_DIR, 'open_prices.json')):
        with open(os.path.join(BASE_DIR, 'open_prices.json'), 'r') as f:
            open_prices = json.load(f)

    if not open_prices:
        print("open_prices.json bulunamadı. Webhook gelmedi mi?")
        return

    latest_prices = pd.Series(open_prices)
    valid_stocks = sig_series.dropna().index.intersection(latest_prices.dropna().index)
    sorted_stocks = sig_series[valid_stocks].sort_values(ascending=False)

    # Önceki gün kapanışlarını bul (Tavan/Taban hesabı için)
    today_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d')
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

    LIMIT_PCT = 0.09
    signal_log_path = os.path.join(BASE_DIR, 'signals_c1_1x1.log')

    # ------------------ LONG SEÇİMİ ------------------
    top_1 = None
    msg_top = "Uygun LONG adayı bulunamadı (Hepsi limit up)."
    for ticker in sorted_stocks.index:
        current_price = float(latest_prices[ticker])
        signal_val = sorted_stocks[ticker]
        
        is_limit_up = False
        if ticker in prev_closes and pd.notna(prev_closes[ticker]):
            prev_close = float(prev_closes[ticker])
            change = (current_price / prev_close) - 1.0
            if change >= LIMIT_PCT:
                is_limit_up = True
                skip_msg = f"[{now_str}] ATLANDI (LONG): {ticker} (Sinyal: {signal_val:+.3f}) -> Fiyat {current_price:.2f} TL önceki kapanışa ({prev_close:.2f} TL) göre +%{change*100:.2f} tavan sınırında!"
                print(skip_msg)
                with open(signal_log_path, 'a', encoding='utf-8') as f:
                    f.write(skip_msg + "\n")
                    
        if not is_limit_up:
            top_1 = ticker
            msg_top = f"Top 1 Long Adayı  : {top_1} (Sinyal: {signal_val:+.3f}, Fiyat: {current_price:.2f} TL)"
            break

    # ------------------ SHORT SEÇİMİ ------------------
    bottom_1 = None
    msg_bot = "Uygun SHORT adayı bulunamadı (Hepsi limit down)."
    for ticker in sorted_stocks.index[::-1]:
        current_price = float(latest_prices[ticker])
        signal_val = sorted_stocks[ticker]
        
        is_limit_down = False
        if ticker in prev_closes and pd.notna(prev_closes[ticker]):
            prev_close = float(prev_closes[ticker])
            change = (current_price / prev_close) - 1.0
            if change <= -LIMIT_PCT:
                is_limit_down = True
                skip_msg = f"[{now_str}] ATLANDI (SHORT): {ticker} (Sinyal: {signal_val:+.3f}) -> Fiyat {current_price:.2f} TL önceki kapanışa ({prev_close:.2f} TL) göre %{change*100:.2f} taban sınırında!"
                print(skip_msg)
                with open(signal_log_path, 'a', encoding='utf-8') as f:
                    f.write(skip_msg + "\n")
                    
        if not is_limit_down:
            bottom_1 = ticker
            msg_bot = f"Bottom 1 Short Adayı: {bottom_1} (Sinyal: {signal_val:+.3f}, Fiyat: {current_price:.2f} TL)"
            break

    print(msg_top)
    print(msg_bot)

    # --- 10-Day Overlapping Logic ---
    history_file = os.path.join(BASE_DIR, 'signals_history_c1.json')
    sig_history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            sig_history = json.load(f)

    # Remove today if it was already run (idempotency)
    sig_history = [s for s in sig_history if s['date'] != today_str]
    
    # Append today's signal
    sig_history.append({
        'date': today_str,
        'long': top_1,
        'short': bottom_1
    })

    # Keep only the last 10 days
    if len(sig_history) > 10:
        sig_history = sig_history[-10:]

    with open(history_file, 'w') as f:
        json.dump(sig_history, f, indent=2)

    # Calculate Target Weights (1/10 for each day)
    target_weights = {}
    n_days_in_history = len(sig_history)
    for s in sig_history:
        l_ticker = s['long']
        s_ticker = s['short']
        if l_ticker:
            target_weights[l_ticker] = target_weights.get(l_ticker, 0.0) + (1.0 / n_days_in_history)
        if s_ticker:
            target_weights[s_ticker] = target_weights.get(s_ticker, 0.0) - (1.0 / n_days_in_history)

    # Mark to Market
    unrealized_pnl = 0.0
    for t, pos in portfolio['active_positions'].items():
        if t in latest_prices:
            cp = float(latest_prices[t])
            pos['curr_price'] = cp
            pnl = (cp - pos['entry_price']) * pos['shares'] if pos['side'] == 'LONG' else (pos['entry_price'] - cp) * pos['shares']
            pos['unrealized_pnl'] = pnl
            unrealized_pnl += pnl

    total_equity = portfolio['cash'] + unrealized_pnl
    portfolio['equity'] = total_equity
    if 'trade_history' not in portfolio: portfolio['trade_history'] = []

    # Process all tickers that are either in current portfolio or in target portfolio
    all_tickers_to_process = set(portfolio['active_positions'].keys()).union(set(target_weights.keys()))

    for t in all_tickers_to_process:
        if t not in latest_prices: continue
        cp = float(latest_prices[t])
        
        target_w = target_weights.get(t, 0.0)
        # BIST doesn't allow fractional shares, we simulate VIOP/Spot with integers
        target_shares = int(abs(target_w) * total_equity / cp) if cp > 0 else 0
        target_side = 'LONG' if target_w > 0 else ('SHORT' if target_w < 0 else None)
        if target_w == 0: target_shares = 0

        # Current state
        pos = portfolio['active_positions'].get(t)
        current_shares = pos['shares'] if pos else 0
        current_side = pos['side'] if pos else None

        # Rebalancing Logic
        if current_shares > 0 and current_side != target_side:
            # Side changed completely, or target is 0. Close current position first.
            realized = (cp - pos['entry_price']) * current_shares if current_side == 'LONG' else (pos['entry_price'] - cp) * current_shares
            cost = (cp * current_shares) * COST_BPS
            portfolio['cash'] += realized - cost
            portfolio['total_realized_pnl'] += realized
            portfolio['total_commission_paid'] += cost
            portfolio['trade_history'].append({
                'ticker': t, 'side': current_side, 'shares': current_shares, 
                'entry_price': pos['entry_price'], 'exit_price': cp, 'realized_pnl': realized, 
                'commission_paid': cost, 'close_time': now_str
            })
            del portfolio['active_positions'][t]
            current_shares = 0
            current_side = None

        if target_shares > 0:
            if current_shares == 0:
                # Open brand new position
                cost = (cp * target_shares) * COST_BPS
                portfolio['cash'] -= cost
                portfolio['total_commission_paid'] += cost
                portfolio['active_positions'][t] = {
                    'side': target_side, 'shares': target_shares, 'entry_price': cp, 'curr_price': cp, 'unrealized_pnl': 0.0
                }
            else:
                # Adjust existing position (same side)
                delta_shares = target_shares - current_shares
                if delta_shares != 0:
                    cost = (cp * abs(delta_shares)) * COST_BPS
                    portfolio['cash'] -= cost
                    portfolio['total_commission_paid'] += cost
                    
                    if delta_shares > 0:
                        # Adding to position: average down entry price
                        old_cost_basis = pos['shares'] * pos['entry_price']
                        new_cost_basis = old_cost_basis + (delta_shares * cp)
                        pos['shares'] = target_shares
                        pos['entry_price'] = new_cost_basis / target_shares
                    else:
                        # Reducing position: realize partial PnL
                        closed_shares = abs(delta_shares)
                        realized = (cp - pos['entry_price']) * closed_shares if current_side == 'LONG' else (pos['entry_price'] - cp) * closed_shares
                        portfolio['cash'] += realized
                        portfolio['total_realized_pnl'] += realized
                        pos['shares'] = target_shares
                        
                        portfolio['trade_history'].append({
                            'ticker': t, 'side': current_side, 'shares': closed_shares, 
                            'entry_price': pos['entry_price'], 'exit_price': cp, 'realized_pnl': realized, 
                            'commission_paid': cost, 'close_time': now_str + " (Partial)"
                        })

    if 'equity_history' not in portfolio: portfolio['equity_history'] = []
    portfolio['equity_history'].append({'time': now_str, 'equity': portfolio['equity']})

    save_portfolio(portfolio)
    print(f"Strateji 4 Güncellendi. Güncel Sanal Özkaynak: {portfolio['equity']:,.2f} TL")


def report():
    p = load_portfolio()
    print("=" * 70)
    print("STRATEJİ 4: Test C1 (1x1) 17:30 REBALANCE PORTFÖY RAPORU")
    print("=" * 70)
    print(f"Son Güncelleme      : {p['last_updated']}")
    print(f"Başlangıç Sermayesi : {p['initial_capital']:,.2f} TL")
    print(f"Güncel Sanal Özkaynak: {p['equity']:,.2f} TL")
    ret_pct = (p['equity'] - p['initial_capital']) / p['initial_capital'] * 100.0
    print(f"Net Toplam Getiri   : {p['equity'] - p['initial_capital']:+,.2f} TL (%{ret_pct:+.2f})")
    print(f"Aktif Pozisyonlar   : {list(p['active_positions'].keys())}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Strateji 3 Botu')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--reset', action='store_true')

    args = parser.parse_args()
    if args.reset:
        reset_portfolio()
    elif args.report:
        report()
    elif args.run:
        run_strategy()
        report()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
