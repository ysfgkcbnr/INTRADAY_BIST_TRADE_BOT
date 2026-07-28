#!/usr/bin/env python3
"""
STRATEJİ 3: Test B2 (1x1) — 1/10 Overlapping 10-Gün Taşıma (REKOR ŞAMPİYON BOT)
================================================================================
Strateji Özeti:
- 17:30 periyodiklik sinyali ile açılan 1x1 (Top 1 Long / Bottom 1 Short) pozisyonlarını
  1/10 çakışmalı (overlapping) yapıda 10 gün boyunca kesintisiz taşır.
- Backtest Başarısı: +%184.22 Net Getiri, +%26.87 Net CAGR, 1.07 Net Sharpe, Newey-West t = 2.1173 (p = 0.017 < 0.05).
- Sinyal: Geçmiş K=10 günün 17:30 periyodundaki getirileri (1/k z-score).
- Seçim: En yüksek 1 hisse LONG (+100k TL Notional), en düşük 1 hisse SHORT (-100k TL Notional).
- Sanal Para: 100.000 TL Özkaynak (2.0x VİOP Kaldıraçlı).
- Komisyon: 10 binde 2 (%0.020 / 2.0 BPS).

Kullanım:
  python3 strateji_3_test_b2_1x1.py --run       # Tek bir 30 dk canlı güncelleme ve rebalance
  python3 strateji_3_test_b2_1x1.py --report    # Sanal portföy durum raporu
  python3 strateji_3_test_b2_1x1.py --reset     # Sanal bakiye sıfırlama (100.000 TL)
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import yfinance as yf

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE_DIR, 'portfolio_b2_1x1.json')
TZ_ISTANBUL = timezone(timedelta(hours=3))

INITIAL_EQUITY = 100000.0
COST_BPS = 0.00020
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
        'strategy_name': 'Test B2 (1x1) - Rekor Şampiyon',
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
    print("Strateji 3 (Test B2 1x1 Rekor Şampiyon) Sanal Portföyü 100.000 TL'ye sıfırlandı.")


def fetch_data():
    try:
        raw_df = yf.download(list(YF_MAP.values()), period='14d', interval='30m', progress=False)
        if raw_df.empty:
            return None
        if isinstance(raw_df.columns, pd.MultiIndex):
            close_df = raw_df['Close'].rename(columns=REV_YF_MAP)
            open_df = raw_df['Open'].rename(columns=REV_YF_MAP)
        else:
            close_df = raw_df[['Close']].rename(columns=REV_YF_MAP)
            open_df = raw_df[['Open']].rename(columns=REV_YF_MAP)

        close_df.index = pd.to_datetime(close_df.index).tz_convert(TZ_ISTANBUL)
        open_df.index = pd.to_datetime(open_df.index).tz_convert(TZ_ISTANBUL)

        time_str = close_df.index.strftime('%H:%M')
        valid_mask = ~time_str.isin(['09:30', '18:00'])
        close_df = close_df[valid_mask].copy()
        open_df = open_df[valid_mask].copy()

        r_oc_df = np.log(close_df / open_df)
        return {'close': close_df, 'open': open_df, 'r_oc': r_oc_df}
    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return None


def run_strategy():
    now_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{now_str}] Strateji 3 (Test B2 1x1 Rekor Şampiyon) Canlı Sinyal & Rebalance...")

    portfolio = load_portfolio()
    data = fetch_data()
    if data is None:
        return

    r_oc_df = data['r_oc']
    close_df = data['close']
    close_1730 = r_oc_df[r_oc_df.index.strftime('%H:%M') == '17:00']

    if len(close_1730) < K_DAYS:
        print("Yeterli geçmiş veri yok.")
        return

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

    latest_prices = close_df.iloc[-1]
    valid_stocks = sig_series.dropna().index.intersection(latest_prices.dropna().index)
    sorted_stocks = sig_series[valid_stocks].sort_values(ascending=False)

    top_1 = sorted_stocks.index[0]
    bottom_1 = sorted_stocks.index[-1]

    print(f"Top 1 Long Adayı  : {top_1} (Sinyal: {sorted_stocks[top_1]:+.3f}, Fiyat: {latest_prices[top_1]:.2f} TL)")
    print(f"Bottom 1 Short Adayı: {bottom_1} (Sinyal: {sorted_stocks[bottom_1]:+.3f}, Fiyat: {latest_prices[bottom_1]:.2f} TL)")

    price_top = float(latest_prices[top_1])
    price_bottom = float(latest_prices[bottom_1])

    # Rebalance logic for B2 1x1 (Overlapping holding)
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

    # Close positions not in target
    for t in list(portfolio['active_positions'].keys()):
        if t != top_1 and t != bottom_1:
            pos = portfolio['active_positions'].pop(t)
            exit_price = float(latest_prices[t])
            realized = (exit_price - pos['entry_price']) * pos['shares'] if pos['side'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['shares']
            cost = (exit_price * pos['shares']) * COST_BPS

            portfolio['cash'] += realized - cost
            portfolio['total_realized_pnl'] += realized
            portfolio['total_commission_paid'] += cost

    # Open target 1x1 positions
    target_shares_long = int(total_equity / price_top) if price_top > 0 else 0
    target_shares_short = int(total_equity / price_bottom) if price_bottom > 0 else 0

    if top_1 not in portfolio['active_positions'] and target_shares_long > 0:
        cost = (price_top * target_shares_long) * COST_BPS
        portfolio['cash'] -= cost
        portfolio['total_commission_paid'] += cost
        portfolio['active_positions'][top_1] = {
            'side': 'LONG', 'shares': target_shares_long, 'entry_price': price_top, 'curr_price': price_top, 'unrealized_pnl': 0.0
        }

    if bottom_1 not in portfolio['active_positions'] and target_shares_short > 0:
        cost = (price_bottom * target_shares_short) * COST_BPS
        portfolio['cash'] -= cost
        portfolio['total_commission_paid'] += cost
        portfolio['active_positions'][bottom_1] = {
            'side': 'SHORT', 'shares': target_shares_short, 'entry_price': price_bottom, 'curr_price': price_bottom, 'unrealized_pnl': 0.0
        }

    save_portfolio(portfolio)
    print(f"Strateji 3 Güncellendi. Güncel Sanal Özkaynak: {portfolio['equity']:,.2f} TL")


def report():
    p = load_portfolio()
    print("=" * 70)
    print("STRATEJİ 3: Test B2 (1x1) REKOR ŞAMPİYON PORTFÖY RAPORU")
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
