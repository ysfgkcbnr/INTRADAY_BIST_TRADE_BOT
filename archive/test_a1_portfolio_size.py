#!/usr/bin/env python3
"""
BIST 100 Intraday Periodicity - Portfolio Concentration Test (N=1, N=2, N=3, N=5)
==================================================================================
Testing user hypothesis: Does focusing on extreme stocks (Top 1 / Bottom 1 or Top 3 / Bottom 3)
improve performance and reduce transaction friction for Test A1 (17:30 bar) and Test B2?

Academic Literature Context (Heston et al. 2010):
- US Universe (~1,500 stocks): Uses Deciles (10% Long / 10% Short) or Quintiles (20% Long / 20% Short).
- BIST Universe (28 stocks):
  * Decile (10%) = Top 3 Long / Bottom 3 Short
  * Quintile (20%) = Top 5 Long / Bottom 5 Short
  * Extreme Signal = Top 1 Long / Bottom 1 Short (1x1)
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

N_SIZES = [1, 2, 3, 5]

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
    """Load BIST 30-min price and return matrix."""
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
# TEST A1 FOR VARIOUS N SIZES
# ============================================================
def run_test_a1_n(ret_pivot, K=10, N=1, cost_bps=0.00020):
    """
    Test A1 (17:30 Bar) with N top/bottom stocks.
    2.0x Leverage (Top N +1.0 notional, Bottom N -1.0 notional).
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

        if j == 16 and d_idx >= K:
            hist_rows = []
            weights = []
            for k in range(1, K + 1):
                prev_d = dates[d_idx - k]
                if (prev_d, 16) in ret_pivot.index:
                    r_k = ret_pivot.loc[(prev_d, 16)].values[0]
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

                if valid_mask.sum() >= (N * 2):
                    valid_indices = np.where(valid_mask)[0]
                    sorted_valid = valid_indices[np.argsort(sig_t[valid_indices])]

                    bottom_indices = sorted_valid[:N]
                    top_indices = sorted_valid[-N:]

                    curr_w = np.zeros(n_stocks)
                    curr_w[top_indices] = 1.0 / N
                    curr_w[bottom_indices] = -1.0 / N
                else:
                    curr_w = np.zeros(n_stocks)
            else:
                curr_w = np.zeros(n_stocks)
        else:
            curr_w = np.zeros(n_stocks)

        ret_t = ret_matrix[t]
        gross_r = np.sum(curr_w * ret_t)

        turnover = np.sum(np.abs(curr_w - prev_w))
        cost = turnover * cost_bps
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

        daily_returns = valid_df.groupby('date')['net_return'].sum()
        win_rate = (daily_returns > 0).mean() * 100.0
        daily_turnover = valid_df.groupby('date')['turnover'].sum().mean() * 100.0

        metrics.append({
            'Strateji Yöntemi': name,
            'Net CAGR (%)': cagr_net * 100.0,
            'Brüt Sharpe': sharpe_gross,
            'Net Sharpe': sharpe_net,
            'Sortino (Net)': sortino_net,
            'Max DD (%)': max_dd * 100.0,
            'Günlük Kazanma (%)': win_rate,
            'Ort. Günlük Turnover (%)': daily_turnover,
            'Net Toplam Getiri (%)': cum_net * 100.0,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, '19_test_a1_portfolio_size_summary.csv'), index=False)
    return metrics_df


def plot_results(results):
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, df in results.items():
        dt_index = pd.to_datetime(df['datetime'])
        c_net = df['cum_net']
        lw = 2.5 if '1x1' in name or '3x3' in name else 1.5
        ax.plot(dt_index, c_net, label=name, linewidth=lw)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}x'))
    ax.set_title('Test A1 (17:30 Barı) Hisse Konsantrasyon Testi (1x1, 2x2, 3x3, 5x5)\n(10 Binde 2 Komisyon, 2.0x VİOP Kaldıraçlı)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '19_test_a1_portfolio_size_equity.png'))
    plt.close()


def main():
    print("=" * 70)
    print("TEST A1 (17:30 BARI) HİSSE KONSANTRASYON MİKTARI TESTİ (N=1, 2, 3, 5)")
    print("=" * 70)

    ret_pivot = load_data()
    results = {}

    # Test N=1, 2, 3, 5 for 2.0 BPS (10 binde 2)
    print("\n--- 10 BİNDE 2 KOMİSYON İLE TEST A1 PORTFÖY BÜYÜKLÜĞÜ HASSASİYETİ ---")
    for N_val in N_SIZES:
        name = f"Test A1 {N_val}x{N_val} ({N_val} Long / {N_val} Short)"
        print(f"Koşturuluyor: {name}...")
        results[name] = run_test_a1_n(ret_pivot, K=10, N=N_val, cost_bps=0.00020)

    metrics_df = calculate_metrics(results)
    plot_results(results)

    print("\n" + "=" * 70)
    print("TEST A1 PORTFÖY BÜYÜKLÜĞÜ PERFORMANS TABLOSU (10 Binde 2 Komisyon)")
    print("=" * 70)
    print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
