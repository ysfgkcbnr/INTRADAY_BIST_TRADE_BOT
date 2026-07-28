#!/usr/bin/env python3
"""
BIST 100 Intraday Periodicity VIOP 2x Leverage Backtest Engine
================================================================
Simulating 100,000 TL Equity with:
- 100,000 TL Spot Long (Top 5 Stocks, 100% of equity)
- 100,000 TL VIOP Short (Bottom 5 Stock Futures, 100% of equity)
- Total Exposure: 200,000 TL (2.0x Leverage on Equity)
- Cost Model: 3.5 BPS (0.035%) per side on traded notional.

Testing K=5 and K=10 across:
1. Test A: Interval-Specific 30-Min Trading (17:30 bar, 10:00 bar, Dual peak)
2. Test B: 1/K Overlapping Multi-Day Holding (5-day and 10-day holding)
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

COST_BPS = 0.0002  # 10 binde 2 (2.0 BPS)
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
# TEST A: VIOP LEVERAGED INTERVAL-SPECIFIC TRADING
# ============================================================
def run_test_a_viop(ret_pivot, target_intervals, K=5):
    """
    Test A: Enter ONLY at target intervals with 2.0x VIOP leverage.
    - 1.0x Equity Spot Long Top 5
    - 1.0x Equity VIOP Short Bottom 5
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

    prev_w = np.zeros(n_stocks)

    for t in range(n_rows):
        d = date_list[t]
        j = intervals[t]
        d_idx = date_to_idx[d]

        if j in target_intervals and d_idx >= K:
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

                    # 2.0x Leverage: Long weights sum to +1.0, Short weights sum to -1.0
                    curr_w = np.zeros(n_stocks)
                    curr_w[top_indices] = 1.0 / N_TOP_BOTTOM
                    curr_w[bottom_indices] = -1.0 / N_TOP_BOTTOM
                else:
                    curr_w = np.zeros(n_stocks)
            else:
                curr_w = np.zeros(n_stocks)
        else:
            curr_w = np.zeros(n_stocks)

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
# TEST B: VIOP LEVERAGED 1/K OVERLAPPING MULTI-DAY HOLDING
# ============================================================
def run_test_b_viop(ret_pivot, target_interval=16, K=5):
    """
    Test B: 1/K Overlapping Multi-day Holding Strategy with 2.0x VIOP leverage.
    """
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    tickers = ret_pivot.columns

    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    timestamps = ret_pivot.index.get_level_values('datetime')
    intervals = ret_pivot.index.get_level_values('interval_idx')
    date_list = ret_pivot.index.get_level_values('date')

    n_days = len(dates)
    sub_weights = np.zeros((n_days, n_stocks))

    for d_idx, d in enumerate(dates):
        if d_idx < K or (d, target_interval) not in ret_pivot.index:
            continue

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

    # Overlapping weights across K days
    overall_day_weights = np.zeros((n_days, n_stocks))
    for d_idx in range(n_days):
        start_d = max(0, d_idx - K + 1)
        overall_day_weights[d_idx] = np.mean(sub_weights[start_d:d_idx + 1], axis=0)

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
# METRICS & PLOTTING
# ============================================================
def calculate_metrics(results):
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
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, '15_viop_leverage_summary.csv'), index=False)
    return metrics_df


def plot_results(results):
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, df in results.items():
        dt_index = pd.to_datetime(df['datetime'])
        c_net = df['cum_net']
        lw = 2.5 if 'Overlapping' in name else 1.5
        ax.plot(dt_index, c_net, label=name, linewidth=lw)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}x'))
    ax.set_title('BIST 100 Spot Long + VIOP Short (2.0x Kaldıraçlı) Net Getiri Eğrileri\n(100 Bin TL Özkaynak ile 100k Spot Buy + 100k VIOP Short, 3.5 BPS Komisyon)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '15_viop_leverage_equity_curves.png'))
    plt.close()


def main():
    print("=" * 70)
    print("BIST VİOP 2.0x KALDIRAÇLI PERİYODİKLİK BACKTEST ENGINE (K=5 ve K=10)")
    print("=" * 70)

    ret_pivot = load_data()
    results = {}

    print("\n--- 1. TEST A: PERİYOT İÇİ 30-DK İŞLEM (SPOT LONG + VIOP SHORT 2.0x) ---")
    print("A1. 17:30 Barı (K=5)...")
    results['Test A1: Sadece 17:30 Barı (K=5)'] = run_test_a_viop(ret_pivot, target_intervals=[16], K=5)
    
    print("A1. 17:30 Barı (K=10)...")
    results['Test A1: Sadece 17:30 Barı (K=10)'] = run_test_a_viop(ret_pivot, target_intervals=[16], K=10)

    print("\n--- 2. TEST B: 1/K OVERLAPPING MULTI-DAY HOLDING (SPOT LONG + VIOP SHORT 2.0x) ---")
    print("B1. 1/5 Overlapping (5-Gün Taşıma, K=5, 17:30 Sinyali)...")
    results['Test B1: 1/5 Overlapping (K=5, 5 Gün Taşıma)'] = run_test_b_viop(ret_pivot, target_interval=16, K=5)

    print("B2. 1/10 Overlapping (10-Gün Taşıma, K=10, 17:30 Sinyali)...")
    results['Test B2: 1/10 Overlapping (K=10, 10 Gün Taşıma)'] = run_test_b_viop(ret_pivot, target_interval=16, K=10)

    print("\n--- PERFORMANS METRİKLERİ HESAPLANIYOR ---")
    metrics_df = calculate_metrics(results)
    plot_results(results)

    print("\n" + "=" * 70)
    print("VİOP 2.0x KALDIRAÇLI BACKTEST PERFORMANS ÖZET TABLOSU")
    print("=" * 70)
    print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
