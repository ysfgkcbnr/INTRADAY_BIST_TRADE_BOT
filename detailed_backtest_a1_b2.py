#!/usr/bin/env python3
"""
BIST 100 Intraday Periodicity - Dedicated Backtest Engine for Test A1 & Test B2
================================================================================
Focused Analysis of:
1. Test A1: 17:30 Closing Bar Trading (Pure 30-min per-day entry/exit)
2. Test B2: 1/10 Overlapping 10-Day Holding (Jegadeesh & Titman 1/K Model)

Leverage Model: 2.0x (100k Spot Buy + 100k VIOP Short on 100k Equity)
Commission Tiers Tested: 0 BPS (Brüt), 2.0 BPS (10 binde 2), 3.5 BPS (10 binde 3.5)

Outputs:
- Monthly & Yearly Return Matrices
- Comprehensive Risk Metrics (Sharpe, Sortino, Calmar, MaxDD, Win Rates, Profit Factor)
- Friction Sensitivity Analysis
- High Resolution Publication Quality Charts
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
import seaborn as sns

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

COMMISSION_TIERS = {
    '0.0 BPS (Brüt)': 0.0,
    '2.0 BPS (10 binde 2)': 0.00020,
    '3.5 BPS (10 binde 3.5)': 0.00035,
}

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
# TEST A1 SIMULATOR
# ============================================================
def run_test_a1(ret_pivot, K=10, cost_bps=0.00020):
    """
    Test A1: 17:30 Closing Bar Trading (K=10).
    Trade ONLY at interval 16 (17:30), 2.0x leverage (Top 5 Long + Bottom 5 Short).
    Liquidate to cash at end of 17:30 bar.
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
# TEST B2 SIMULATOR
# ============================================================
def run_test_b2(ret_pivot, K=10, cost_bps=0.00020):
    """
    Test B2: 1/10 Overlapping 10-Day Holding Strategy (K=10).
    - Signal generated at interval 16 (17:30).
    - Position held across all 16 intervals for 10 trading days.
    - Rebalanced in 1/10 overlapping sub-portfolios each day.
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
        if d_idx < K or (d, 16) not in ret_pivot.index:
            continue

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
            sig_d = np.tensordot(w_arr, np.array(hist_rows), axes=(0, 0))

            r_d = ret_pivot.loc[(d, 16)].values[0]
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
# STATISTICAL & RISK METRICS ENGINE
# ============================================================
def compute_detailed_metrics(results_dict):
    """Compute exhaustive performance, risk, and friction metrics."""
    metrics = []
    intervals_per_year = 252 * INTERVALS_PER_DAY

    for strat_name, df in results_dict.items():
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

        # Drawdowns
        cum_series = np.exp(np.cumsum(valid_df['net_return']))
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak
        max_dd = dd.min()

        # Calmar Ratio
        calmar = cagr_net / abs(max_dd) if abs(max_dd) > 0 else np.nan

        # Daily win rate
        daily_returns = valid_df.groupby('date')['net_return'].sum()
        daily_win_rate = (daily_returns > 0).mean() * 100.0

        # Profit Factor
        pos_sum = daily_returns[daily_returns > 0].sum()
        neg_sum = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = pos_sum / neg_sum if neg_sum > 0 else np.nan

        # Turnover & Cost Paid
        daily_turnover = valid_df.groupby('date')['turnover'].sum().mean() * 100.0
        total_turnover = valid_df['turnover'].sum() * 100.0

        metrics.append({
            'Strateji': strat_name,
            'Net CAGR (%)': cagr_net * 100.0,
            'Brüt Sharpe': sharpe_gross,
            'Net Sharpe': sharpe_net,
            'Sortino (Net)': sortino_net,
            'Calmar Oranı': calmar,
            'Max DD (%)': max_dd * 100.0,
            'Günlük Kazanma (%)': daily_win_rate,
            'Profit Factor': profit_factor,
            'Ort. Günlük Turnover (%)': daily_turnover,
            'Net Toplam Getiri (%)': cum_net * 100.0,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, '17_detailed_metrics_summary.csv'), index=False)
    return metrics_df


# ============================================================
# MONTHLY HEATMAP GENERATOR FOR TEST B2
# ============================================================
def generate_monthly_heatmap(df_b2_net):
    """Generate monthly return matrix for Test B2."""
    df_b2_net['date'] = pd.to_datetime(df_b2_net['date'])
    daily_ret = df_b2_net.groupby('date')['net_return'].sum()
    
    monthly_ret = daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df = monthly_ret.unstack(level=1)
    
    month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                   'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    monthly_df.columns = [month_names[c - 1] for c in monthly_df.columns]

    # Calculate Annual Totals
    yearly_ret = daily_ret.groupby(daily_ret.index.year).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df['Yıllık Toplam'] = yearly_ret

    monthly_df.to_csv(os.path.join(OUTPUT_DIR, '17_test_b2_monthly_returns.csv'))

    # Heatmap Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(monthly_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=True, ax=ax, linewidths=0.5)
    ax.set_title('Test B2 (1/10 Overlapping 10-Gün Taşıma) Aylık Net Getiri Matrisi (%)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Yıl')
    ax.set_xlabel('Ay')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '17_test_b2_monthly_heatmap.png'))
    plt.close()

    return monthly_df


# ============================================================
# VISUALIZATIONS
# ============================================================
def plot_detailed_backtest(results_dict):
    """Generate professional comparative plots for Test A1 and Test B2."""
    # 1. Net Equity Curves Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    styles = {
        'Test B2 (2.0 BPS)': ('#1f77b4', 2.5, '-'),
        'Test B2 (3.5 BPS)': ('#aec7e8', 1.8, '--'),
        'Test B2 (Brüt - 0.0 BPS)': ('#2ca02c', 1.8, ':'),
        'Test A1 (2.0 BPS)': ('#ff7f0e', 2.0, '-'),
        'Test A1 (3.5 BPS)': ('#d62728', 1.5, '--'),
        'Test A1 (Brüt - 0.0 BPS)': ('#9467bd', 2.5, '-'),
    }

    for name, df in results_dict.items():
        dt_index = pd.to_datetime(df['datetime'])
        c_net = df['cum_net']
        color, lw, ls = styles.get(name, ('#7f7f7f', 1.5, '-'))
        ax.plot(dt_index, c_net, label=name, color=color, linewidth=lw, linestyle=ls)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}x'))
    ax.set_title('BIST 100 Detaylı Backtest: Test A1 (17:30 Barı) vs Test B2 (1/10 Overlapping)\n(2.0x VİOP Kaldıraçlı, Farklı Komisyon Katmanları)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '17_detailed_equity_curves.png'))
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("=" * 70)
    print("BIST 100 DETAYLI TEST A1 VE TEST B2 BACKTEST ENGINE")
    print("======================================================================")

    ret_pivot = load_data()
    results = {}

    print("\n--- 1. TEST A1 (Sadece 17:30 Barı, K=10) Komisyon Hassasiyeti ---")
    for tier_name, bps in COMMISSION_TIERS.items():
        name = f"Test A1 ({tier_name})"
        print(f"Koşturuluyor: {name}...")
        results[name] = run_test_a1(ret_pivot, K=10, cost_bps=bps)

    print("\n--- 2. TEST B2 (1/10 Overlapping 10-Gün Taşıma) Komisyon Hassasiyeti ---")
    for tier_name, bps in COMMISSION_TIERS.items():
        name = f"Test B2 ({tier_name})"
        print(f"Koşturuluyor: {name}...")
        results[name] = run_test_b2(ret_pivot, K=10, cost_bps=bps)

    print("\n--- DETAYLI PERFORMANS & RİSK METRİKLERİ HESAPLANIYOR ---")
    metrics_df = compute_detailed_metrics(results)
    
    # Generate Monthly Heatmap for Test B2 (2.0 BPS)
    monthly_matrix = generate_monthly_heatmap(results['Test B2 (2.0 BPS (10 binde 2))'])

    # Plots
    plot_detailed_backtest(results)

    print("\n" + "=" * 70)
    print("DETAYLI BACKTEST PERFORMANS & RİSK METRİKLERİ TABLOSU")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TEST B2 (2.0 BPS) AYLIK NET GETİRİ MATRİSİ (%)")
    print("=" * 70)
    print(monthly_matrix.to_string())


if __name__ == '__main__':
    main()
