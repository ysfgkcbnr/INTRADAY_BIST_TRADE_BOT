#!/usr/bin/env python3
"""
BIST 100 Intraday & Overnight Periodicity Backtest Engine
==========================================================
Testing Overnight Holding Strategy:
- Entry: Date d at 17:00 (17:30 closing bar opening)
- Hold: Overnight through market close to next morning
- Exit: Date d+1 at 10:30 (Opening 30-min bar completion)

Total Held Return = 17:00-17:30 Return (Day d) + Overnight Gap (Day d to d+1) + 10:00-10:30 Return (Day d+1)

Leverage: 2.0x (100k Spot Buy Top 5 + 100k VIOP Short Bottom 5 on 100k Equity)
Costs: 3.5 BPS and 2.0 BPS single side tested.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TZ_ISTANBUL = timezone(timedelta(hours=3))
INTERVALS_PER_DAY = 16
EXCLUDED_TIMES = {'09:30', '18:00'}

N_TOP_BOTTOM = 5

# Plot style
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """Load and prepare 30-min BIST data matrix."""
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'BISTMIXED_*.csv')))
    all_frames = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        ticker = fname.replace('BISTMIXED_', '').split(',')[0].split(' ')[0]
        df = pd.read_csv(fpath)
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(TZ_ISTANBUL)
        df['date'] = df['datetime'].dt.date
        df['time_str'] = df['datetime'].dt.strftime('%H:%M')
        df['ticker'] = ticker
        df = df.drop_duplicates(subset=['time'], keep='first')
        df = df[~df['time_str'].isin(EXCLUDED_TIMES)].copy()
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    day_counts = combined.groupby(['ticker', 'date']).size().unstack(fill_value=0)
    full_days = day_counts.columns[(day_counts == INTERVALS_PER_DAY).all(axis=0)]
    combined = combined[combined['date'].isin(full_days)].copy()

    combined = combined.sort_values(['date', 'time_str', 'ticker'])
    time_to_idx = {t: i + 1 for i, t in enumerate(sorted(combined['time_str'].unique()))}
    combined['interval_idx'] = combined['time_str'].map(time_to_idx)

    combined['r_oc'] = np.log(combined['close'] / combined['open'])
    
    # Pivot for r_oc
    ret_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='r_oc')

    # Pivot for Open and Close prices to calculate full overnight return
    open_pivot = combined.pivot_table(index=['date', 'interval_idx'], columns='ticker', values='open')
    close_pivot = combined.pivot_table(index=['date', 'interval_idx'], columns='ticker', values='close')

    return ret_pivot, open_pivot, close_pivot


# ============================================================
# OVERNIGHT STRATEGY BACKTEST ENGINE
# ============================================================
def run_overnight_strategy(ret_pivot, open_pivot, close_pivot, K=10, cost_bps=0.00035):
    """
    Overnight Strategy:
    - Entry at 17:00 (Interval 16) on Day d using K-day 17:30 periodicity signal.
    - Held Overnight through 17:30 close, overnight gap, to 10:30 close on Day d+1.
    - Total return per stock = ln(Close_{d+1, 1} / Open_{d, 16})
    """
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    tickers = ret_pivot.columns

    n_days = len(dates)
    n_stocks = len(tickers)

    daily_net_returns = []
    daily_gross_returns = []
    daily_turnovers = []
    trade_dates = []

    prev_w = np.zeros(n_stocks)

    for i in range(K, n_days - 1):
        d_curr = dates[i]
        d_next = dates[i + 1]

        # Calculate periodicity signal at 17:30 (Interval 16) over past K days
        hist_rows = []
        weights = []
        for k in range(1, K + 1):
            prev_d = dates[i - k]
            if (prev_d, 16) in ret_pivot.index:
                r_k = ret_pivot.loc[(prev_d, 16)].values[0]
                valid_m = np.isfinite(r_k)
                if valid_m.sum() > 5:
                    z_k = (r_k - np.nanmean(r_k)) / (np.nanstd(r_k) + 1e-8)
                    hist_rows.append(z_k)
                    weights.append(1.0 / k)

        if not hist_rows:
            continue

        w_arr = np.array(weights) / np.sum(weights)
        sig_d = np.tensordot(w_arr, np.array(hist_rows), axes=(0, 0))

        # Check entry prices on day d_curr (17:00 Open, Interval 16)
        p_entry = open_pivot.loc[(d_curr, 16)].values
        # Check exit prices on day d_next (10:30 Close, Interval 1)
        p_exit = close_pivot.loc[(d_next, 1)].values

        # Stock return over the held period: ln(p_exit / p_entry)
        held_return = np.log(p_exit / p_entry)
        valid_mask = np.isfinite(sig_d) & np.isfinite(held_return)

        if valid_mask.sum() >= (N_TOP_BOTTOM * 2):
            valid_indices = np.where(valid_mask)[0]
            sorted_valid = valid_indices[np.argsort(sig_d[valid_indices])]

            bottom_indices = sorted_valid[:N_TOP_BOTTOM]
            top_indices = sorted_valid[-N_TOP_BOTTOM:]

            # 2.0x Leverage: Top 5 +1.0, Bottom 5 -1.0
            curr_w = np.zeros(n_stocks)
            curr_w[top_indices] = 1.0 / N_TOP_BOTTOM
            curr_w[bottom_indices] = -1.0 / N_TOP_BOTTOM

            # Gross return of held position
            gross_r = np.sum(curr_w * held_return)

            # Turnover: opening new position on day d_curr, closing on day d_next
            turnover = np.sum(np.abs(curr_w - prev_w)) + np.sum(np.abs(curr_w))
            cost = turnover * cost_bps
            net_r = gross_r - cost

            daily_gross_returns.append(gross_r)
            daily_net_returns.append(net_r)
            daily_turnovers.append(turnover)
            trade_dates.append(d_next)
        else:
            prev_w = np.zeros(n_stocks)

    df_res = pd.DataFrame({
        'date': trade_dates,
        'gross_return': daily_gross_returns,
        'net_return': daily_net_returns,
        'turnover': daily_turnovers,
    })
    df_res['cum_gross'] = np.exp(np.cumsum(df_res['gross_return']))
    df_res['cum_net'] = np.exp(np.cumsum(df_res['net_return']))

    return df_res


# ============================================================
# METRICS & MAIN EXECUTION
# ============================================================
def calculate_metrics(results):
    metrics = []
    days_per_year = 252

    for name, df in results.items():
        n_periods = len(df)
        years = n_periods / days_per_year

        cum_gross = np.exp(df['gross_return'].sum()) - 1.0
        cum_net = np.exp(df['net_return'].sum()) - 1.0

        cagr_gross = (1.0 + cum_gross) ** (1.0 / max(years, 0.1)) - 1.0
        cagr_net = (1.0 + cum_net) ** (1.0 / max(years, 0.1)) - 1.0

        vol_gross = df['gross_return'].std() * np.sqrt(days_per_year)
        vol_net = df['net_return'].std() * np.sqrt(days_per_year)

        sharpe_gross = cagr_gross / vol_gross if vol_gross > 0 else np.nan
        sharpe_net = cagr_net / vol_net if vol_net > 0 else np.nan

        downside_net = df[df['net_return'] < 0]['net_return']
        downside_vol = downside_net.std() * np.sqrt(days_per_year) if len(downside_net) > 0 else np.nan
        sortino_net = cagr_net / downside_vol if downside_vol > 0 else np.nan

        cum_series = np.exp(np.cumsum(df['net_return']))
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak
        max_dd = dd.min()

        win_rate = (df['net_return'] > 0).mean() * 100.0
        daily_turnover = df['turnover'].mean() * 100.0

        metrics.append({
            'Strateji Yöntemi': name,
            'Net CAGR (%)': cagr_net * 100.0,
            'Brüt Sharpe': sharpe_gross,
            'Net Sharpe': sharpe_net,
            'Sortino (Net)': sortino_net,
            'Max DD (%)': max_dd * 100.0,
            'Kazanma Oranı (%)': win_rate,
            'Ort. Günlük Turnover (%)': daily_turnover,
            'Net Toplam Getiri (%)': cum_net * 100.0,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, '16_overnight_strategy_summary.csv'), index=False)
    return metrics_df


def plot_results(results):
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, df in results.items():
        dt_index = pd.to_datetime(df['date'])
        c_net = df['cum_net']
        ax.plot(dt_index, c_net, label=name, linewidth=2.0)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}x'))
    ax.set_title('BIST 100 Gece Taşıma Stratejisi (17:00 Giriş -> Gece Taşı -> Ertesi Sabah 10:30 Çıkış)\n(100k Spot Buy + 100k VIOP Short, 2.0x Kaldıraçlı)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '16_overnight_equity_curves.png'))
    plt.close()


def main():
    print("=" * 70)
    print("BIST GECE TAŞIMA PERİYODİKLİK STRATEJİSİ BACKTEST ENGINE")
    print("======================================================================")

    ret_pivot, open_pivot, close_pivot = load_data()
    results = {}

    print("\n1. 3.5 BPS Komisyon İle Gece Taşıma (K=5 ve K=10)...")
    results['Gece Taşıma K=5 (3.5 BPS)'] = run_overnight_strategy(ret_pivot, open_pivot, close_pivot, K=5, cost_bps=0.00035)
    results['Gece Taşıma K=10 (3.5 BPS)'] = run_overnight_strategy(ret_pivot, open_pivot, close_pivot, K=10, cost_bps=0.00035)

    print("\n2. 2.0 BPS (10 Binde 2) Komisyon İle Gece Taşıma (K=5 ve K=10)...")
    results['Gece Taşıma K=5 (2.0 BPS)'] = run_overnight_strategy(ret_pivot, open_pivot, close_pivot, K=5, cost_bps=0.00020)
    results['Gece Taşıma K=10 (2.0 BPS)'] = run_overnight_strategy(ret_pivot, open_pivot, close_pivot, K=10, cost_bps=0.00020)

    metrics_df = calculate_metrics(results)
    plot_results(results)

    print("\n" + "=" * 70)
    print("GECE TAŞIMA STRATEJİSİ PERFORMANS ÖZET TABLOSU")
    print("=" * 70)
    print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
