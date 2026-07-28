#!/usr/bin/env python3
"""
BIST 100 Intraday Periodicity Literature Backtest Engine
=========================================================
Testing Academic Literature Methods (Heston, Korajczyk & Sadka, 2010):

Test A: Interval-Specific 30-Min Trading (Heston et al. 2010 Section IV)
  - Trade ONLY high significance intervals:
    * A1: 17:30 (Closing 30-min bar)
    * A2: 10:00 (Opening 30-min bar)
    * A3: 10:00 & 17:30 (Dual Peak intervals)
  - Hold for 30 minutes, liquidate to cash. Zero overnight exposure outside interval.

Test B: 1/K Overlapping Multi-Day Holding (Jegadeesh & Titman 1/K Model)
  - Signal at peak interval (17:30).
  - Position held for K trading days (K=5 and K=10).
  - 1/K overlapping sub-portfolio rebalancing to minimize turnover.

Cost Model: 3.5 BPS (0.035%) per side.
Portfolio: Top 5 Long / Bottom 5 Short (Market Neutral).
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
INTERVALS_PER_DAY = 16  # Tradeable 30-min intervals per day (10:00-17:30)
EXCLUDED_TIMES = {'09:30', '18:00'}

COST_BPS = 0.00035  # 10 binde 3.5
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
    ret_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='r_oc')

    return ret_pivot


# ============================================================
# TEST A: INTERVAL-SPECIFIC 30-MIN TRADING (HESTON 2010)
# ============================================================
def run_test_a_interval_specific(ret_pivot, target_intervals, K=10, name_suffix=""):
    """
    Test A: Enter ONLY at specific high-significance 30-min intervals.
    Hold for 30 minutes, exit to cash.
    """
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    tickers = ret_pivot.columns

    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    timestamps = ret_pivot.index.get_level_values('datetime')
    intervals = ret_pivot.index.get_level_values('interval_idx')
    date_list = ret_pivot.index.get_level_values('date')

    gross_returns = np.zeros(n_rows)
    net_returns = np.zeros(n_rows)
    turnovers = np.zeros(n_rows)
    weights_mat = np.zeros((n_rows, n_stocks))

    prev_w = np.zeros(n_stocks)

    for t in range(n_rows):
        d = date_list[t]
        j = intervals[t]
        d_idx = date_to_idx[d]

        # Check if current interval is a target trading interval
        if j in target_intervals and d_idx >= K:
            # Calculate periodicity signal for interval j using past K days at interval j
            hist_rows = []
            weights = []
            for k in range(1, K + 1):
                prev_d = dates[d_idx - k]
                if (prev_d, j) in ret_pivot.index:
                    r_k = ret_pivot.loc[(prev_d, j)].values[0]
                    valid_m = np.isfinite(r_k)
                    if valid_m.sum() > 5:
                        z_k = (r_k - np.nanmean(r_k)) / (np.nanstd(r_k) + 1e-8)
                        hist_rows.append(z_k)
                        weights.append(1.0 / k)

            if hist_rows:
                w_arr = np.array(weights) / np.sum(weights)
                sig_t = np.tensordot(w_arr, np.array(hist_rows), axes=(0, 0))

                ret_t = ret_matrix[t]
                valid_mask = np.isfinite(sig_t) & np.isfinite(ret_t)

                if valid_mask.sum() >= (N_TOP_BOTTOM * 2):
                    valid_indices = np.where(valid_mask)[0]
                    sorted_valid = valid_indices[np.argsort(sig_t[valid_indices])]

                    bottom_indices = sorted_valid[:N_TOP_BOTTOM]
                    top_indices = sorted_valid[-N_TOP_BOTTOM:]

                    curr_w = np.zeros(n_stocks)
                    curr_w[top_indices] = 1.0 / N_TOP_BOTTOM
                    curr_w[bottom_indices] = -1.0 / N_TOP_BOTTOM
                else:
                    curr_w = np.zeros(n_stocks)
            else:
                curr_w = np.zeros(n_stocks)
        else:
            # Not a target interval -> hold ZERO position
            curr_w = np.zeros(n_stocks)

        ret_t = ret_matrix[t]
        gross_r = np.sum(curr_w * ret_t)

        # Turnover calculation (change from prev_w to curr_w)
        turnover = np.sum(np.abs(curr_w - prev_w))
        cost = turnover * COST_BPS
        net_r = gross_r - cost

        gross_returns[t] = gross_r
        net_returns[t] = net_r
        turnovers[t] = turnover
        weights_mat[t] = curr_w
        prev_w = curr_w

    df_res = pd.DataFrame({
        'datetime': timestamps,
        'date': date_list,
        'interval_idx': intervals,
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnovers,
    })
    df_res['cum_gross'] = np.exp(np.cumsum(gross_returns))
    df_res['cum_net'] = np.exp(np.cumsum(net_returns))

    return df_res


# ============================================================
# TEST B: 1/K OVERLAPPING MULTI-DAY HOLDING (JEGADEESH & TITMAN)
# ============================================================
def run_test_b_overlapping(ret_pivot, target_interval=16, K=5):
    """
    Test B: 1/K Overlapping Multi-day Holding Strategy.
    - Signal generated at target_interval (e.g. 17:30).
    - Sub-portfolio formed each day is held for K trading days.
    - 1/K of the portfolio is rebalanced each day.
    """
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    tickers = ret_pivot.columns

    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    timestamps = ret_pivot.index.get_level_values('datetime')
    intervals = ret_pivot.index.get_level_values('interval_idx')
    date_list = ret_pivot.index.get_level_values('date')

    # Array of sub-portfolio target weights for each day d
    n_days = len(dates)
    sub_weights = np.zeros((n_days, n_stocks))

    for d_idx, d in enumerate(dates):
        if d_idx < K or (d, target_interval) not in ret_pivot.index:
            continue

        # Get periodicity signal at target_interval for past K days
        hist_rows = []
        weights = []
        for k in range(1, K + 1):
            prev_d = dates[d_idx - k]
            if (prev_d, target_interval) in ret_pivot.index:
                r_k = ret_pivot.loc[(prev_d, target_interval)].values[0]
                valid_m = np.isfinite(r_k)
                if valid_m.sum() > 5:
                    z_k = (r_k - np.nanmean(r_k)) / (np.nanstd(r_k) + 1e-8)
                    hist_rows.append(z_k)
                    weights.append(1.0 / k)

        if hist_rows:
            w_arr = np.array(weights) / np.sum(weights)
            sig_d = np.tensordot(w_arr, np.array(hist_rows), axes=(0, 0))

            r_d = ret_pivot.loc[(d, target_interval)].values[0]
            valid_mask = np.isfinite(sig_d) & np.isfinite(r_d)

            if valid_mask.sum() >= (N_TOP_BOTTOM * 2):
                valid_indices = np.where(valid_mask)[0]
                sorted_valid = valid_indices[np.argsort(sig_d[valid_indices])]

                bottom_indices = sorted_valid[:N_TOP_BOTTOM]
                top_indices = sorted_valid[-N_TOP_BOTTOM:]

                w_sub = np.zeros(n_stocks)
                w_sub[top_indices] = 1.0 / N_TOP_BOTTOM
                w_sub[bottom_indices] = -1.0 / N_TOP_BOTTOM
                sub_weights[d_idx] = w_sub

    # Construct 1/K Overlapping Portfolio Weights:
    # Portfolio weight on day d is the mean of sub-portfolios from day d-K+1 to d
    overall_day_weights = np.zeros((n_days, n_stocks))
    for d_idx in range(n_days):
        start_d = max(0, d_idx - K + 1)
        overall_day_weights[d_idx] = np.mean(sub_weights[start_d:d_idx + 1], axis=0)

    # Apply overall_day_weights across all 16 intervals of day d
    gross_returns = np.zeros(n_rows)
    net_returns = np.zeros(n_rows)
    turnovers = np.zeros(n_rows)

    prev_w = np.zeros(n_stocks)

    for t in range(n_rows):
        d = date_list[t]
        d_idx = date_to_idx[d]
        curr_w = overall_day_weights[d_idx]

        ret_t = ret_matrix[t]
        gross_r = np.sum(curr_w * ret_t)

        turnover = np.sum(np.abs(curr_w - prev_w))
        cost = turnover * COST_BPS
        net_r = gross_r - cost

        gross_returns[t] = gross_r
        net_returns[t] = net_r
        turnovers[t] = turnover
        prev_w = curr_w

    df_res = pd.DataFrame({
        'datetime': timestamps,
        'date': date_list,
        'interval_idx': intervals,
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnovers,
    })
    df_res['cum_gross'] = np.exp(np.cumsum(gross_returns))
    df_res['cum_net'] = np.exp(np.cumsum(net_returns))

    return df_res


# ============================================================
# METRICS COMPUTE & PLOTTING
# ============================================================
def calculate_metrics(results):
    """Calculate CAGR, Sharpe, Sortino, MaxDD and Turnover."""
    metrics = []
    intervals_per_year = 252 * INTERVALS_PER_DAY

    for name, df in results.items():
        valid_df = df[df['net_return'] != 0.0]
        if len(valid_df) == 0:
            valid_df = df

        n_periods = len(valid_df)
        years = n_periods / intervals_per_year

        cum_gross = np.exp(valid_df['gross_return'].sum()) - 1.0
        cum_net = np.exp(valid_df['net_return'].sum()) - 1.0

        cagr_gross = (1.0 + cum_gross) ** (1.0 / max(years, 0.1)) - 1.0
        cagr_net = (1.0 + cum_net) ** (1.0 / max(years, 0.1)) - 1.0

        vol_gross = valid_df['gross_return'].std() * np.sqrt(intervals_per_year)
        vol_net = valid_df['net_return'].std() * np.sqrt(intervals_per_year)

        sharpe_gross = cagr_gross / vol_gross if vol_gross > 0 else np.nan
        sharpe_net = cagr_net / vol_net if vol_net > 0 else np.nan

        downside_net = valid_df[valid_df['net_return'] < 0]['net_return']
        downside_vol = downside_net.std() * np.sqrt(intervals_per_year) if len(downside_net) > 0 else np.nan
        sortino_net = cagr_net / downside_vol if downside_vol > 0 else np.nan

        cum_series = np.exp(np.cumsum(valid_df['net_return']))
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak
        max_dd = dd.min()

        daily_turnover = valid_df.groupby('date')['turnover'].sum().mean() * 100.0

        metrics.append({
            'Strateji Yöntemi': name,
            'Net CAGR (%)': cagr_net * 100.0,
            'Brüt Sharpe': sharpe_gross,
            'Net Sharpe': sharpe_net,
            'Sortino (Net)': sortino_net,
            'Max DD (%)': max_dd * 100.0,
            'Ort. Günlük Turnover (%)': daily_turnover,
            'Net Toplam Getiri (%)': cum_net * 100.0,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, '14_literature_backtest_summary.csv'), index=False)
    return metrics_df


def plot_results(results):
    """Plot cumulative net returns comparing Test A and Test B methods."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, df in results.items():
        dt_index = pd.to_datetime(df['datetime'])
        c_net = df['cum_net']
        lw = 2.5 if '17:30' in name or 'Overlapping' in name else 1.5
        ax.plot(dt_index, c_net, label=name, linewidth=lw)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}x'))
    ax.set_title('BIST 100 Literatür Yöntemleri Karşılaştırmalı Net Getiri Eğrileri\n(Test A: 30-Dk Periyot İşlemi vs. Test B: 1/K Overlapping Taşıma, 3.5 BPS Komisyon)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '14_literature_equity_curves.png'))
    plt.close()


def main():
    print("=" * 70)
    print("BIST LİTERATÜR GÜN İÇİ PERİYODİKLİK BACKTEST ENGINE")
    print("=" * 70)

    ret_pivot = load_data()
    results = {}

    print("\n--- TEST A: PERİYOT İÇİ 30-DK İŞLEM (HESTON ET AL. 2010 SECTION IV) ---")
    print("A1. Sadece 17:30 Kapanış Barı (K=10)...")
    results['Test A1: Sadece 17:30 Barı (K=10)'] = run_test_a_interval_specific(ret_pivot, target_intervals=[16], K=10)

    print("A2. Sadece 10:00 Açılış Barı (K=10)...")
    results['Test A2: Sadece 10:00 Barı (K=10)'] = run_test_a_interval_specific(ret_pivot, target_intervals=[1], K=10)

    print("A3. Çift Pik Barı (10:00 & 17:30, K=10)...")
    results['Test A3: Çift Bar (10:00 + 17:30, K=10)'] = run_test_a_interval_specific(ret_pivot, target_intervals=[1, 16], K=10)

    print("\n--- TEST B: 1/K OVERLAPPING MULTI-DAY HOLDING (JEGADEESH & TITMAN) ---")
    print("B1. 1/5 Overlapping 5-Gün Taşıma (17:30 Sinyali)...")
    results['Test B1: 1/5 Overlapping (5 Gün Taşıma)'] = run_test_b_overlapping(ret_pivot, target_interval=16, K=5)

    print("B2. 1/10 Overlapping 10-Gün Taşıma (17:30 Sinyali)...")
    results['Test B2: 1/10 Overlapping (10 Gün Taşıma)'] = run_test_b_overlapping(ret_pivot, target_interval=16, K=10)

    print("\n--- PERFORMANS METRİKLERİ HESAPLANIYOR ---")
    metrics_df = calculate_metrics(results)
    plot_results(results)

    print("\n" + "=" * 70)
    print("LİTERATÜR BACKTEST PERFORMANS ÖZET TABLOSU")
    print("=" * 70)
    print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
