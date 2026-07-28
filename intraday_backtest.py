#!/usr/bin/env python3
"""
BIST 100 Intraday Periodicity & Short-Term Reversal Backtest Engine
====================================================================
Based on Heston, Korajczyk & Sadka (2010, Journal of Finance)
"Intraday Patterns in the Cross-Section of Stock Returns"

Features:
- Long-Short Market Neutral Portfolio Construction (Top 5 Long / Bottom 5 Short)
- 30-minute Rebalancing (16 tradeable intervals per day: 10:00 - 17:30)
- Realistic Transaction Costs: 3.5 BPS (0.035% per side)
- Multiple Strategy Signals:
  1. Periodicity_K5 (Past 5-day same 30-min bar weighted signal)
  2. Periodicity_K10 (Past 10-day same 30-min bar weighted signal)
  3. ShortTermReversal (Immediate previous 30-min bar reversal)
  4. Composite_K5_Rev (50% Periodicity K5 + 50% Reversal)
  5. Composite_K10_Rev (50% Periodicity K10 + 50% Reversal)
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

# Transaction Cost: 10 binde 3.5 = 3.5 bps = 0.00035
COST_BPS = 0.00035

# Portfolio Selection: Top N Long, Bottom N Short
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
# DATA LOADING & PREPARATION
# ============================================================
def load_and_prepare_data():
    """Load BIST CSVs and construct aligned price and return matrices."""
    print("=" * 70)
    print("VERİ YÜKLEME VE MATRİS HAZIRLIĞI")
    print("=" * 70)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'BISTMIXED_*.csv')))
    print(f"{len(csv_files)} adet hisse CSV dosyası okunuyor...")

    all_frames = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        ticker = fname.replace('BISTMIXED_', '').split(',')[0].split(' ')[0]

        df = pd.read_csv(fpath)
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(TZ_ISTANBUL)
        df['date'] = df['datetime'].dt.date
        df['time_str'] = df['datetime'].dt.strftime('%H:%M')
        df['ticker'] = ticker

        # Duplicate temizliği ve 09:30 / 18:00 filtresi
        df = df.drop_duplicates(subset=['time'], keep='first')
        df = df[~df['time_str'].isin(EXCLUDED_TIMES)].copy()
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    # Tam gün filtreleme (16 periyot)
    day_counts = combined.groupby(['ticker', 'date']).size().unstack(fill_value=0)
    full_days = day_counts.columns[(day_counts == INTERVALS_PER_DAY).all(axis=0)]
    combined = combined[combined['date'].isin(full_days)].copy()

    # Periyot indeksleme (1 to 16)
    combined = combined.sort_values(['date', 'time_str', 'ticker'])
    time_to_idx = {t: i + 1 for i, t in enumerate(sorted(combined['time_str'].unique()))}
    combined['interval_idx'] = combined['time_str'].map(time_to_idx)

    # Logaritmik getiriler
    combined['r_oc'] = np.log(combined['close'] / combined['open'])
    
    # Pivot tablolar: Index = (date, interval_idx, datetime), Columns = ticker
    ret_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='r_oc')
    
    print(f"Toplam İşlem Günü: {len(full_days)}")
    print(f"Toplam 30-Dk Periyot Sayısı: {len(ret_pivot)}")
    print(f"Hisse Sayısı: {ret_pivot.shape[1]}")
    
    return ret_pivot, combined


# ============================================================
# SIGNAL GENERATORS
# ============================================================
def compute_signals(ret_pivot):
    """
    Compute cross-sectionally standardized signals for each strategy:
    1. Periodicity K=5 (1/k decaying weights)
    2. Periodicity K=10 (1/k decaying weights)
    3. Short-Term Reversal (previous 30-min bar)
    4. Composite K=5 + Rev
    5. Composite K=10 + Rev
    """
    print("\n" + "=" * 70)
    print("SİNYAL ÜRETİMİ HESAPLANIYOR")
    print("=" * 70)

    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    tickers = ret_pivot.columns

    n_rows = len(ret_pivot)
    n_stocks = len(tickers)

    sig_per5 = np.full((n_rows, n_stocks), np.nan)
    sig_per10 = np.full((n_rows, n_stocks), np.nan)
    sig_rev = np.full((n_rows, n_stocks), np.nan)

    # 1 & 2. Periodicity Signals (K=5 and K=10)
    print("-> Gün içi periyodiklik sinyalleri hesaplanıyor (K=5 ve K=10)...")
    for (d, j, dt), row in ret_pivot.iterrows():
        d_idx = date_to_idx[d]
        row_pos = ret_pivot.index.get_loc((d, j, dt))

        # K=5 periodicity
        if d_idx >= 5:
            hist_rows_5 = []
            weights_5 = []
            for k in range(1, 6):
                prev_d = dates[d_idx - k]
                if (prev_d, j) in ret_pivot.index:
                    r_k = ret_pivot.loc[(prev_d, j)].values[0]
                    valid_m = np.isfinite(r_k)
                    if valid_m.sum() > 5:
                        z_k = (r_k - np.nanmean(r_k)) / (np.nanstd(r_k) + 1e-8)
                        hist_rows_5.append(z_k)
                        weights_5.append(1.0 / k)

            if hist_rows_5:
                w_arr = np.array(weights_5) / np.sum(weights_5)
                sig_per5[row_pos, :] = np.tensordot(w_arr, np.array(hist_rows_5), axes=(0, 0))

        # K=10 periodicity
        if d_idx >= 10:
            hist_rows_10 = []
            weights_10 = []
            for k in range(1, 11):
                prev_d = dates[d_idx - k]
                if (prev_d, j) in ret_pivot.index:
                    r_k = ret_pivot.loc[(prev_d, j)].values[0]
                    valid_m = np.isfinite(r_k)
                    if valid_m.sum() > 5:
                        z_k = (r_k - np.nanmean(r_k)) / (np.nanstd(r_k) + 1e-8)
                        hist_rows_10.append(z_k)
                        weights_10.append(1.0 / k)

            if hist_rows_10:
                w_arr = np.array(weights_10) / np.sum(weights_10)
                sig_per10[row_pos, :] = np.tensordot(w_arr, np.array(hist_rows_10), axes=(0, 0))

    # 3. Short-Term Reversal Signal (Lag-1 30-min bar)
    print("-> Short-Term Reversal sinyali hesaplanıyor (Lag-1 bar)...")
    ret_values = ret_pivot.values
    for idx in range(1, n_rows):
        prev_r = ret_values[idx - 1]
        valid_m = np.isfinite(prev_r)
        if valid_m.sum() > 5:
            z_prev = (prev_r - np.nanmean(prev_r)) / (np.nanstd(prev_r) + 1e-8)
            # Negative sign for reversal (underperformed -> Long, outperformed -> Short)
            sig_rev[idx, :] = -z_prev

    # 4 & 5. Composite Signals
    print("-> Kompozit sinyaller birleştiriliyor...")
    sig_comp5 = np.full((n_rows, n_stocks), np.nan)
    sig_comp10 = np.full((n_rows, n_stocks), np.nan)

    valid_c5 = np.isfinite(sig_per5) & np.isfinite(sig_rev)
    sig_comp5[valid_c5] = 0.5 * sig_per5[valid_c5] + 0.5 * sig_rev[valid_c5]

    valid_c10 = np.isfinite(sig_per10) & np.isfinite(sig_rev)
    sig_comp10[valid_c10] = 0.5 * sig_per10[valid_c10] + 0.5 * sig_rev[valid_c10]

    signals = {
        'Periodicity (K=5)': sig_per5,
        'Periodicity (K=10)': sig_per10,
        'Short-Term Reversal': sig_rev,
        'Composite (K=5 + Rev)': sig_comp5,
        'Composite (K=10 + Rev)': sig_comp10,
    }

    return signals


# ============================================================
# BACKTEST SIMULATION ENGINE (LONG-SHORT)
# ============================================================
def run_backtest(ret_pivot, signals, n_top=N_TOP_BOTTOM, cost_bps=COST_BPS):
    """
    Simulate 30-min Long-Short Rebalancing:
    - Top N stocks: Long (+1/N weight)
    - Bottom N stocks: Short (-1/N weight)
    - Rebalanced every 30-minute interval
    - Transaction cost applied on turnover change
    """
    print("\n" + "=" * 70)
    print(f"BACKTEST SİMÜLASYONU BAŞLATILIYOR (Long-Short, Top/Bottom {n_top}, Maliyet={cost_bps*10000:.1f} bps)")
    print("=" * 70)

    ret_matrix = ret_pivot.values  # (T, S)
    timestamps = ret_pivot.index.get_level_values('datetime')
    intervals = ret_pivot.index.get_level_values('interval_idx')
    dates = ret_pivot.index.get_level_values('date')
    n_rows, n_stocks = ret_matrix.shape

    results = {}

    for strat_name, sig_matrix in signals.items():
        print(f"Simülasyon koşturuluyor: {strat_name}...")

        weights = np.zeros((n_rows, n_stocks))
        gross_returns = np.zeros(n_rows)
        net_returns = np.zeros(n_rows)
        turnovers = np.zeros(n_rows)

        prev_w = np.zeros(n_stocks)

        for t in range(n_rows):
            sig_t = sig_matrix[t]
            ret_t = ret_matrix[t]

            valid_mask = np.isfinite(sig_t) & np.isfinite(ret_t)
            n_valid = valid_mask.sum()

            if n_valid >= (n_top * 2):
                valid_indices = np.where(valid_mask)[0]
                sorted_valid = valid_indices[np.argsort(sig_t[valid_indices])]

                bottom_indices = sorted_valid[:n_top]  # Short
                top_indices = sorted_valid[-n_top:]   # Long

                curr_w = np.zeros(n_stocks)
                curr_w[top_indices] = 1.0 / n_top
                curr_w[bottom_indices] = -1.0 / n_top
            else:
                curr_w = np.zeros(n_stocks)

            gross_r = np.sum(curr_w * ret_t)

            turnover = np.sum(np.abs(curr_w - prev_w))
            cost = turnover * cost_bps
            net_r = gross_r - cost

            gross_returns[t] = gross_r
            net_returns[t] = net_r
            turnovers[t] = turnover
            weights[t] = curr_w
            prev_w = curr_w

        df_res = pd.DataFrame({
            'datetime': timestamps,
            'date': dates,
            'interval_idx': intervals,
            'gross_return': gross_returns,
            'net_return': net_returns,
            'turnover': turnovers,
        })
        df_res['cum_gross'] = np.exp(np.cumsum(gross_returns))
        df_res['cum_net'] = np.exp(np.cumsum(net_returns))

        results[strat_name] = {
            'df': df_res,
            'weights': weights,
        }

    # Benchmark: Equal Weighted Long-Only Buy & Hold across all stocks
    print("Benchmark hesaplanıyor (BIST 28 Hisse Eşit Ağırlık Buy&Hold)...")
    bench_returns = np.nanmean(ret_matrix, axis=1)
    bench_df = pd.DataFrame({
        'datetime': timestamps,
        'date': dates,
        'interval_idx': intervals,
        'gross_return': bench_returns,
        'net_return': bench_returns,
        'turnover': 0.0,
    })
    bench_df['cum_gross'] = np.exp(np.cumsum(bench_returns))
    bench_df['cum_net'] = bench_df['cum_gross']
    results['Benchmark (Eşit Ağırlık)'] = {'df': bench_df, 'weights': None}

    return results


# ============================================================
# METRICS & STATISTICAL ANALYSIS
# ============================================================
def calculate_performance_metrics(results):
    """Calculate comprehensive performance metrics for all strategies."""
    metrics_list = []

    intervals_per_year = 252 * INTERVALS_PER_DAY  # 4032 intervals / year

    for name, res in results.items():
        df = res['df']
        
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

        win_rate_30m = (valid_df['net_return'] > 0).mean() * 100.0
        
        daily_net = valid_df.groupby('date')['net_return'].sum()
        win_rate_daily = (daily_net > 0).mean() * 100.0

        daily_turnover = valid_df.groupby('date')['turnover'].sum().mean() * 100.0

        metrics_list.append({
            'Strateji': name,
            'Net CAGR (%)': cagr_net * 100.0,
            'Brüt Sharpe': sharpe_gross,
            'Net Sharpe': sharpe_net,
            'Sortino (Net)': sortino_net,
            'Max DD (%)': max_dd * 100.0,
            '30-Dk Kazanma (%)': win_rate_30m,
            'Günlük Kazanma (%)': win_rate_daily,
            'Ort. Günlük Turnover (%)': daily_turnover,
            'Net Toplam Getiri (%)': cum_net * 100.0,
        })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, '13_backtest_summary.csv'), index=False)
    
    return metrics_df


# ============================================================
# INTERVAL PERFORMANCE BREAKDOWN
# ============================================================
def calculate_interval_breakdown(results):
    """Analyze which 30-min intervals (10:00 - 17:30) contribute most to profits."""
    interval_performance = []

    for name, res in results.items():
        if 'Benchmark' in name:
            continue
        df = res['df']
        grouped = df.groupby('interval_idx')['net_return'].agg(['mean', 'std', 'count'])
        grouped['sharpe'] = (grouped['mean'] / grouped['std']) * np.sqrt(252)
        grouped['strategy'] = name
        interval_performance.append(grouped.reset_index())

    if interval_performance:
        all_int_df = pd.concat(interval_performance, ignore_index=True)
        all_int_df.to_csv(os.path.join(OUTPUT_DIR, '13_interval_performance_breakdown.csv'), index=False)
        return all_int_df
    return None


# ============================================================
# PLOTTING & VISUALIZATION
# ============================================================
def plot_backtest_results(results, metrics_df, int_df):
    """Generate high quality charts for backtest equity curves and interval breakdowns."""
    print("\n" + "=" * 70)
    print("PERFORMANS GRAFİKLERİ OLUŞTURULUYOR")
    print("=" * 70)

    # 1. Equity Curves Plot (Net Performance)
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        'Periodicity (K=5)': '#1f77b4',
        'Periodicity (K=10)': '#aec7e8',
        'Short-Term Reversal': '#ff7f0e',
        'Composite (K=5 + Rev)': '#2ca02c',
        'Composite (K=10 + Rev)': '#d62728',
        'Benchmark (Eşit Ağırlık)': '#7f7f7f',
    }

    for name, res in results.items():
        df = res['df']
        dt_index = pd.to_datetime(df['datetime'])
        c_net = df['cum_net']
        color = colors.get(name, None)
        lw = 2.5 if 'Composite' in name or 'Benchmark' in name else 1.5
        ls = '--' if 'Benchmark' in name else '-'
        ax.plot(dt_index, c_net, label=name, color=color, linewidth=lw, linestyle=ls)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}x'))
    ax.set_title('BIST 100 Gün İçi Stratejiler - Net Kumulatif Getiri Eğrileri (Log Ölçek)\n(Long-Short Market Neutral, 10 Binde 3.5 Komisyon)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '13_backtest_equity_curves.png'))
    plt.close()

    # 2. Drawdown Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, res in results.items():
        if name == 'Benchmark (Eşit Ağırlık)':
            continue
        df = res['df']
        dt_index = pd.to_datetime(df['datetime'])
        cum_series = df['cum_net']
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak * 100.0
        ax.plot(dt_index, dd, label=name, linewidth=1.2, alpha=0.85)

    ax.set_title('Stratejilerin Drawdown (Zirveden Düşüş %) Profili', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Tarih')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '13_backtest_drawdowns.png'))
    plt.close()

    # 3. Interval Breakdown Bar Chart
    if int_df is not None:
        fig, ax = plt.subplots(figsize=(14, 6))
        time_labels = ['10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
                       '13:00', '13:30', '14:00', '14:30', '15:00', '15:30',
                       '16:00', '16:30', '17:00', '17:30']
        
        pivot_int = int_df.pivot(index='interval_idx', columns='strategy', values='mean') * 10000.0  # In bps
        
        plot_strats = [s for s in pivot_int.columns if 'Composite' in s or 'Reversal' in s]
        pivot_int[plot_strats].plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xticklabels(time_labels[:len(pivot_int)], rotation=45)
        ax.set_title('30-Dakikalık Zaman Dilimlerine Göre Ortalama Net Getiri (BPS)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Gün İçi 30-Dakikalık Zaman Dilimi')
        ax.set_ylabel('Ortalama Net Getiri (BPS = 0.01%)')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '13_backtest_interval_performance.png'))
        plt.close()

    print("Tüm grafikler 'output/' dizinine başarıyla kaydedildi.")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    start_time = datetime.now()
    print("=" * 70)
    print("BIST GÜN İÇİ PERİYODİKLİK & REVERSAL BACKTEST SİMÜLATÖRÜ")
    print(f"Başlangıç Zamanı: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    ret_pivot, combined_df = load_and_prepare_data()

    signals = compute_signals(ret_pivot)

    results = run_backtest(ret_pivot, signals, n_top=N_TOP_BOTTOM, cost_bps=COST_BPS)

    metrics_df = calculate_performance_metrics(results)

    int_df = calculate_interval_breakdown(results)

    plot_backtest_results(results, metrics_df, int_df)

    print("\n" + "=" * 70)
    print("BACKTEST PERFORMANS ÖZET TABLOSU")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nİşlem başarıyla tamamlandı! Toplam süre: {duration:.2f} saniye.")


if __name__ == '__main__':
    main()
