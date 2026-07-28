#!/usr/bin/env python3
"""
BIST 100 Intraday Periodicity - Live Paper Trading System (paper_trader.py)
=============================================================================
Real-time Paper Trading Bot & Virtual Portfolio Manager using Yahoo Finance API.

Default Champion Strategy: Test B2 (1x1)
- 1/10 Overlapping 10-Day Holding Model
- Top 1 Stock Spot Long (+100k TL Notional)
- Bottom 1 Stock VIOP Short (-100k TL Notional)
- Virtual Capital: 100,000 TL Equity (2.0x Leverage)
- Transaction Cost: 2.0 BPS (10 binde 2)

Usage:
  python3 paper_trader.py --run       # Run single 30-min live update & rebalance
  python3 paper_trader.py --daemon    # Run continuous 30-min market loop
  python3 paper_trader.py --report    # Display virtual portfolio dashboard
  python3 paper_trader.py --reset     # Reset virtual portfolio to 100,000 TL
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import yfinance as yf

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE_DIR, 'paper_portfolio.json')
TZ_ISTANBUL = timezone(timedelta(hours=3))

INITIAL_EQUITY = 100000.0  # 100,000 TL Virtual Capital
COST_BPS = 0.00020        # 10 binde 2 (2.0 BPS)
LEVERAGE = 2.0            # 2.0x Notional Exposure (100k Long + 100k Short)

TICKERS_BIST = [
    'AEFES', 'AKBNK', 'ASELS', 'BIMAS', 'EKGYO', 'ENKAI', 'EREGL', 'FROTO',
    'GARAN', 'GUBRF', 'ISCTR', 'KCHOL', 'KRDMD', 'MGROS', 'PETKM', 'PGSUS',
    'SAHOL', 'SASA', 'SISE', 'TAVHL', 'TCELL', 'THYAO', 'TOASO', 'TRALT',
    'TTKOM', 'TUPRS', 'VAKBN', 'YKBNK'
]

YF_MAP = {t: f"{t}.IS" for t in TICKERS_BIST}
REV_YF_MAP = {v: k for k, v in YF_MAP.items()}


# ============================================================
# PORTFOLIO STATE PERSISTENCE (paper_portfolio.json)
# ============================================================
def load_portfolio():
    """Load paper portfolio state from JSON file or create initial state."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Uyarı: Portfolio dosyası okunamadı, yeniden oluşturuluyor ({e}).")

    # Initial Portfolio Structure
    portfolio = {
        'created_at': datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S'),
        'last_updated': datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S'),
        'initial_capital': INITIAL_EQUITY,
        'cash': INITIAL_EQUITY,
        'equity': INITIAL_EQUITY,
        'total_realized_pnl': 0.0,
        'total_unrealized_pnl': 0.0,
        'total_commission_paid': 0.0,
        'active_positions': {},  # ticker -> {side: 'LONG'/'SHORT', shares, entry_price, curr_price, unrealized_pnl}
        'trade_history': [],    # list of executed paper trades
        'equity_history': []    # list of {timestamp, equity, drawdown}
    }
    save_portfolio(portfolio)
    return portfolio


def save_portfolio(portfolio):
    """Save paper portfolio state to JSON file."""
    portfolio['last_updated'] = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def reset_portfolio():
    """Reset virtual portfolio to 100,000 TL initial capital."""
    if os.path.exists(PORTFOLIO_FILE):
        os.remove(PORTFOLIO_FILE)
    p = load_portfolio()
    print("Sanal Portföy 100.000 TL başlangıç sermayesine başarıyla sıfırlandı.")
    return p


# ============================================================
# YAHOO FINANCE INTRADAY DATA FETCHING
# ============================================================
def fetch_live_30m_data():
    """Fetch 30m intraday data for all BIST stocks via yfinance."""
    print("Yahoo Finance API üzerinden 30 dakikalık canlı veriler çekiliyor...")
    yf_tickers = list(YF_MAP.values())

    try:
        raw_df = yf.download(yf_tickers, period='14d', interval='30m', progress=False)
    except Exception as e:
        print(f"Hata: yfinance veri indirmesi başarısız oldu ({e}).")
        return None

    if raw_df.empty:
        print("Hata: yfinance boş veri döndürdü.")
        return None

    # Handle multi-level columns from yfinance
    if isinstance(raw_df.columns, pd.MultiIndex):
        close_df = raw_df['Close'].rename(columns=REV_YF_MAP)
        open_df = raw_df['Open'].rename(columns=REV_YF_MAP)
    else:
        close_df = raw_df[['Close']].rename(columns=REV_YF_MAP)
        open_df = raw_df[['Open']].rename(columns=REV_YF_MAP)

    # Convert Datetime index to Istanbul Timezone
    close_df.index = pd.to_datetime(close_df.index).tz_convert(TZ_ISTANBUL)
    open_df.index = pd.to_datetime(open_df.index).tz_convert(TZ_ISTANBUL)

    # Filter out non-tradeable auction bars (09:30 and 18:00)
    time_str = close_df.index.strftime('%H:%M')
    valid_mask = ~time_str.isin(['09:30', '18:00'])
    close_df = close_df[valid_mask].copy()
    open_df = open_df[valid_mask].copy()

    # Log returns r_oc = ln(Close / Open)
    r_oc_df = np.log(close_df / open_df)

    return {
        'close': close_df,
        'open': open_df,
        'r_oc': r_oc_df
    }


# ============================================================
# SIGNAL GENERATOR (TEST B2 1x1 & TEST A1 1x1)
# ============================================================
def compute_live_signals(data_dict, K=10):
    """
    Compute 17:30 periodicity signal across past K=10 trading days.
    Returns: top_1_stock (Long), bottom_1_stock (Short), signal_series
    """
    r_oc_df = data_dict['r_oc']
    close_df = data_dict['close']

    # Filter to 17:30 closing bar entries (17:00 time_str)
    close_1730 = r_oc_df[r_oc_df.index.strftime('%H:%M') == '17:00'].copy()

    if len(close_1730) < K:
        print(f"Uyarı: Yeterli geçmiş 17:30 periyot verisi yok (Mevcut: {len(close_1730)}, Gerekli: {K}).")
        return None, None, None

    # Take last K trading days at 17:30
    recent_k = close_1730.iloc[-K:]

    hist_zscores = []
    weights = []

    for k in range(1, K + 1):
        r_k = recent_k.iloc[-k].values
        valid_m = np.isfinite(r_k)
        if valid_m.sum() > 5:
            z_k = (r_k - np.nanmean(r_k)) / (np.nanstd(r_k) + 1e-8)
            hist_zscores.append(z_k)
            weights.append(1.0 / k)

    if not hist_zscores:
        return None, None, None

    w_arr = np.array(weights) / np.sum(weights)
    sig_series = pd.Series(np.tensordot(w_arr, np.array(hist_zscores), axes=(0, 0)), index=r_oc_df.columns)

    # Latest prices
    latest_prices = close_df.iloc[-1]
    valid_stocks = sig_series.dropna().index.intersection(latest_prices.dropna().index)

    sorted_stocks = sig_series[valid_stocks].sort_values(ascending=False)

    top_1_stock = sorted_stocks.index[0]      # Best Long
    bottom_1_stock = sorted_stocks.index[-1]   # Best Short

    return top_1_stock, bottom_1_stock, sig_series


# ============================================================
# LIVE PAPER REBALANCING ENGINE
# ============================================================
def run_live_rebalance(portfolio, mode='B2_1x1'):
    """Fetch live data, calculate signal, rebalance paper portfolio."""
    now_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{now_str}] Live Sanal Rebalance İşlemi Başlatılıyor...")

    data_dict = fetch_live_30m_data()
    if data_dict is None:
        print("Canlı veri alınamadı. İşlem atlanıyor.")
        return portfolio

    top_1, bottom_1, sig_series = compute_live_signals(data_dict, K=10)
    if top_1 is None or bottom_1 is None:
        print("Sinyal üretilemedi.")
        return portfolio

    latest_prices = data_dict['close'].iloc[-1]
    price_top = float(latest_prices[top_1])
    price_bottom = float(latest_prices[bottom_1])

    print(f"\n--- CANLI SİNYAL SONUÇLARI ---")
    print(f"Top 1 Long Adayı  : {top_1} (Fiyat: {price_top:.2f} TL, Sinyal Skor: {sig_series[top_1]:+.3f})")
    print(f"Bottom 1 Short Adayı: {bottom_1} (Fiyat: {price_bottom:.2f} TL, Sinyal Skor: {sig_series[bottom_1]:+.3f})")

    # Update current position prices & unrealized PnL
    curr_equity = portfolio['cash']
    unrealized_pnl = 0.0

    for ticker, pos in portfolio['active_positions'].items():
        if ticker in latest_prices:
            cp = float(latest_prices[ticker])
            pos['curr_price'] = cp
            if pos['side'] == 'LONG':
                pnl = (cp - pos['entry_price']) * pos['shares']
            else:
                pnl = (pos['entry_price'] - cp) * pos['shares']
            pos['unrealized_pnl'] = pnl
            unrealized_pnl += pnl

    total_equity = portfolio['cash'] + unrealized_pnl
    portfolio['equity'] = total_equity
    portfolio['total_unrealized_pnl'] = unrealized_pnl

    # Target Capital Allocation (100k TL Notional Long + 100k TL Notional Short)
    target_long_notional = total_equity * 1.0
    target_short_notional = total_equity * 1.0

    target_long_shares = int(target_long_notional / price_top) if price_top > 0 else 0
    target_short_shares = int(target_short_notional / price_bottom) if price_bottom > 0 else 0

    # Rebalance logic: check if current positions match target
    curr_long = [t for t, p in portfolio['active_positions'].items() if p['side'] == 'LONG']
    curr_short = [t for t, p in portfolio['active_positions'].items() if p['side'] == 'SHORT']

    # 1. Close positions that are no longer in target
    for t in curr_long:
        if t != top_1:
            pos = portfolio['active_positions'].pop(t)
            exit_price = float(latest_prices[t])
            realized = (exit_price - pos['entry_price']) * pos['shares']
            cost = (exit_price * pos['shares']) * COST_BPS

            portfolio['cash'] += (exit_price * pos['shares']) - cost
            portfolio['total_realized_pnl'] += realized
            portfolio['total_commission_paid'] += cost

            portfolio['trade_history'].append({
                'timestamp': now_str,
                'ticker': t,
                'action': 'SELL_LONG',
                'shares': pos['shares'],
                'price': exit_price,
                'realized_pnl': realized,
                'commission': cost
            })
            print(f"[REBALANCE] LONG Pozisyon Kapatıldı: {t} @ {exit_price:.2f} TL (Kâr/Zarar: {realized:+.2f} TL)")

    for t in curr_short:
        if t != bottom_1:
            pos = portfolio['active_positions'].pop(t)
            exit_price = float(latest_prices[t])
            realized = (pos['entry_price'] - exit_price) * pos['shares']
            cost = (exit_price * pos['shares']) * COST_BPS

            portfolio['cash'] += realized - cost
            portfolio['total_realized_pnl'] += realized
            portfolio['total_commission_paid'] += cost

            portfolio['trade_history'].append({
                'timestamp': now_str,
                'ticker': t,
                'action': 'COVER_SHORT',
                'shares': pos['shares'],
                'price': exit_price,
                'realized_pnl': realized,
                'commission': cost
            })
            print(f"[REBALANCE] SHORT Pozisyon Kapatıldı: {t} @ {exit_price:.2f} TL (Kâr/Zarar: {realized:+.2f} TL)")

    # 2. Open or adjust target positions
    if top_1 not in portfolio['active_positions'] and target_long_shares > 0:
        cost = (price_top * target_long_shares) * COST_BPS
        portfolio['cash'] -= cost
        portfolio['total_commission_paid'] += cost

        portfolio['active_positions'][top_1] = {
            'side': 'LONG',
            'shares': target_long_shares,
            'entry_price': price_top,
            'curr_price': price_top,
            'unrealized_pnl': 0.0
        }
        portfolio['trade_history'].append({
            'timestamp': now_str,
            'ticker': top_1,
            'action': 'BUY_LONG',
            'shares': target_long_shares,
            'price': price_top,
            'commission': cost
        })
        print(f"[REBALANCE] Yeni LONG Pozisyon Açıldı: {top_1} -> {target_long_shares} Adet @ {price_top:.2f} TL")

    if bottom_1 not in portfolio['active_positions'] and target_short_shares > 0:
        cost = (price_bottom * target_short_shares) * COST_BPS
        portfolio['cash'] -= cost
        portfolio['total_commission_paid'] += cost

        portfolio['active_positions'][bottom_1] = {
            'side': 'SHORT',
            'shares': target_short_shares,
            'entry_price': price_bottom,
            'curr_price': price_bottom,
            'unrealized_pnl': 0.0
        }
        portfolio['trade_history'].append({
            'timestamp': now_str,
            'ticker': bottom_1,
            'action': 'OPEN_SHORT',
            'shares': target_short_shares,
            'price': price_bottom,
            'commission': cost
        })
        print(f"[REBALANCE] Yeni SHORT Pozisyon Açıldı: {bottom_1} -> {target_short_shares} Adet @ {price_bottom:.2f} TL")

    # Update equity history
    eq_series = [e['equity'] for e in portfolio['equity_history']] + [portfolio['equity']]
    peak = max(eq_series)
    dd = (portfolio['equity'] - peak) / peak * 100.0

    portfolio['equity_history'].append({
        'timestamp': now_str,
        'equity': portfolio['equity'],
        'drawdown_pct': dd
    })

    save_portfolio(portfolio)
    print(f"[{now_str}] Rebalance Başarıyla Tamamlandı. Güncel Sanal Özkaynak: {portfolio['equity']:,.2f} TL")
    return portfolio


# ============================================================
# CLI REPORT DASHBOARD
# ============================================================
def display_portfolio_report(portfolio):
    """Print high quality CLI report of virtual portfolio state."""
    print("=" * 70)
    print("BIST 100 GÜN İÇİ PERİYODİKLİK - CANLI SANAL PORTFÖY RAPORU")
    print("=" * 70)
    print(f"Son Güncelleme      : {portfolio['last_updated']}")
    print(f"Başlangıç Sermayesi : {portfolio['initial_capital']:,.2f} TL")
    print(f"Güncel Sanal Özkaynak: {portfolio['equity']:,.2f} TL")
    
    total_ret_pct = (portfolio['equity'] - portfolio['initial_capital']) / portfolio['initial_capital'] * 100.0
    print(f"Net Toplam Kâr/Zarar: {portfolio['equity'] - portfolio['initial_capital']:+,.2f} TL (%{total_ret_pct:+.2f})")
    print(f"Gerçekleşen PnL     : {portfolio['total_realized_pnl']:+,.2f} TL")
    print(f"Açık Pozisyon PnL   : {portfolio['total_unrealized_pnl']:+,.2f} TL")
    print(f"Ödenen Sanal Komisyon: {portfolio['total_commission_paid']:,.2f} TL (10 Binde 2)")

    print("\n" + "-" * 70)
    print("AKTİF POZİSYONLAR (Top 1 Long / Bottom 1 Short)")
    print("-" * 70)
    if not portfolio['active_positions']:
        print("Şu anda aktif açık pozisyon bulunmuyor.")
    else:
        pos_df = pd.DataFrame.from_dict(portfolio['active_positions'], orient='index')
        pos_df = pos_df.reset_index().rename(columns={'index': 'Hisse'})
        print(pos_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("SON İŞLEM GEÇMİŞİ (Son 5 İşlem)")
    print("-" * 70)
    if not portfolio['trade_history']:
        print("Henüz gerçekleşmiş işlem bulunmuyor.")
    else:
        recent_trades = pd.DataFrame(portfolio['trade_history'][-5:])
        print(recent_trades.to_string(index=False))

    print("=" * 70)


# ============================================================
# MAIN CLI ENTRYPOINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='BIST 100 Live Paper Trader Bot')
    parser.add_argument('--run', action='store_true', help='Execute single 30-min live update & rebalance')
    parser.add_argument('--daemon', action='store_true', help='Run continuous 30-min market loop daemon')
    parser.add_argument('--report', action='store_true', help='Display current virtual portfolio report')
    parser.add_argument('--reset', action='store_true', help='Reset virtual portfolio to 100,000 TL initial capital')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.reset:
        reset_portfolio()
        return

    portfolio = load_portfolio()

    if args.report:
        display_portfolio_report(portfolio)
        return

    if args.run:
        portfolio = run_live_rebalance(portfolio, mode='B2_1x1')
        display_portfolio_report(portfolio)
        return

    if args.daemon:
        print("=" * 70)
        print("PAPER TRADER DAEMON BAŞLATILDI (30 Dakikalık Canlı Döngü)")
        print("Durdurmak için Ctrl+C tuşlarına basabilirsiniz.")
        print("=" * 70)
        while True:
            try:
                now = datetime.now(TZ_ISTANBUL)
                hour = now.hour
                minute = now.minute

                # Run during market hours 10:00 to 18:00 Istanbul time
                if 10 <= hour <= 18:
                    portfolio = run_live_rebalance(portfolio, mode='B2_1x1')
                    display_portfolio_report(portfolio)

                print("\n[DAEMON] 30 dakika boyunca bekleniyor...")
                time.sleep(1800)  # Sleep 30 minutes (1800s)
            except KeyboardInterrupt:
                print("\n[DAEMON] Kullanıcı tarafından durduruldu.")
                break
            except Exception as e:
                print(f"[DAEMON HATA]: {e}. 60 saniye sonra tekrar denenecek...")
                time.sleep(60)


if __name__ == '__main__':
    main()
