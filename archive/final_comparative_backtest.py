#!/usr/bin/env python3
"""
BIST 100 Intraday Periodicity - Final Master Comparative Engine
================================================================
Head-to-Head Comparison of 3 Winning Strategies:
1. Test A1 (1x1): 17:30 Closing Bar, Top 1 Long / Bottom 1 Short
2. Test B2 (5x5): 1/10 Overlapping 10-Day Holding, Top 5 Long / Bottom 5 Short
3. Test B2 (1x1): 1/10 Overlapping 10-Day Holding, Top 1 Long / Bottom 1 Short

Rigorous Testing Applied to ALL 3 Strategies:
- Full Performance & Risk Metrics (CAGR, Sharpe, Sortino, Calmar, MaxDD, Win Rate, Profit Factor)
- Newey-West HAC t-Test (Statistical Hypothesis Testing)
- CSCV PBO (Probability of Backtest Overfitting)
- Nested Walk-Forward Optimization (Rolling OOS Test)
- Monthly & Yearly Return Heatmaps & Tables for each strategy
"""

import os
import glob
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
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
COST_BPS = 0.00035  # 10 binde 3.5 default (3.5 BPS)

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
# SIMULATOR FUNCTIONS
# ============================================================
def run_test_a1(ret_pivot, K=10, N=1, cost_bps=COST_BPS):
    """Test A1 Simulator for given N."""
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

    daily_df = df_res.groupby('date').agg({'gross_return': 'sum', 'net_return': 'sum', 'turnover': 'sum'}).reset_index()

    return df_res, daily_df


def run_test_b2(ret_pivot, K=10, N=5, cost_bps=COST_BPS):
    """Test B2 Simulator for given N."""
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

            if valid_mask.sum() >= (N * 2):
                valid_indices = np.where(valid_mask)[0]
                sorted_valid = valid_indices[np.argsort(sig_d[valid_indices])]

                bottom_indices = sorted_valid[:N]
                top_indices = sorted_valid[-N:]

                w_sub = np.zeros(n_stocks)
                w_sub[top_indices] = 1.0 / N
                w_sub[bottom_indices] = -1.0 / N
                sub_weights[d_idx] = w_sub

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

    daily_df = df_res.groupby('date').agg({'gross_return': 'sum', 'net_return': 'sum', 'turnover': 'sum'}).reset_index()

    return df_res, daily_df


# ============================================================
# STATISTICAL TEST ENGINES
# ============================================================
def compute_metrics(strat_name, df_res, daily_df):
    """Compute complete metric suite for a strategy."""
    intervals_per_year = 252 * INTERVALS_PER_DAY
    valid_df = df_res[df_res['net_return'] != 0.0]

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

    calmar = cagr_net / abs(max_dd) if abs(max_dd) > 0 else np.nan

    daily_net = daily_df['net_return']
    win_rate = (daily_net > 0).mean() * 100.0

    pos_sum = daily_net[daily_net > 0].sum()
    neg_sum = abs(daily_net[daily_net < 0].sum())
    profit_factor = pos_sum / neg_sum if neg_sum > 0 else np.nan

    daily_turnover = daily_df['turnover'].mean() * 100.0

    return {
        'Strateji': strat_name,
        'Net CAGR (%)': cagr_net * 100.0,
        'Brüt Sharpe': sharpe_gross,
        'Net Sharpe': sharpe_net,
        'Sortino (Net)': sortino_net,
        'Calmar Oranı': calmar,
        'Max DD (%)': max_dd * 100.0,
        'Günlük Kazanma (%)': win_rate,
        'Profit Factor': profit_factor,
        'Ort. Günlük Turnover (%)': daily_turnover,
        'Net Toplam Getiri (%)': cum_net * 100.0,
    }


def compute_newey_west(daily_df, max_lag=10):
    """Compute Newey-West t-test."""
    series = daily_df['net_return'].dropna().values
    T = len(series)
    X = np.ones((T, 1))
    model = sm.OLS(series, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})

    mean_ret = model.params[0]
    se = model.bse[0]
    t_stat = model.tvalues[0]
    p_val = 1.0 - sp_stats.t.cdf(t_stat, df=T-1) if t_stat > 0 else 1.0

    return t_stat, p_val


def compute_pbo_cscv(ret_pivot, strat_type='b2', N_val=5, S=16):
    """Compute PBO score for specific strategy type."""
    K_grid = [3, 5, 7, 10, 12, 15]
    grid_daily = []

    for K_val in K_grid:
        if strat_type == 'a1':
            _, d_df = run_test_a1(ret_pivot, K=K_val, N=N_val, cost_bps=COST_BPS)
        else:
            _, d_df = run_test_b2(ret_pivot, K=K_val, N=N_val, cost_bps=COST_BPS)
        grid_daily.append(d_df['net_return'].values)

    daily_returns_matrix = np.column_stack(grid_daily)
    T, N_params = daily_returns_matrix.shape
    block_size = T // S

    matrix_clean = daily_returns_matrix[:S * block_size, :]
    blocks = [matrix_clean[i * block_size:(i + 1) * block_size, :] for i in range(S)]

    combos = list(itertools.combinations(range(S), S // 2))
    is_best_oos_ranks = []

    for combo in combos:
        is_idx = list(combo)
        oos_idx = [i for i in range(S) if i not in is_idx]

        is_matrix = np.vstack([blocks[i] for i in is_idx])
        oos_matrix = np.vstack([blocks[i] for i in oos_idx])

        is_sharpes = np.mean(is_matrix, axis=0) / (np.std(is_matrix, axis=0) + 1e-8)
        best_is_param = np.argmax(is_sharpes)

        oos_sharpes = np.mean(oos_matrix, axis=0) / (np.std(oos_matrix, axis=0) + 1e-8)
        oos_ranks = pd.Series(oos_sharpes).rank(pct=True).values
        is_best_oos_ranks.append(oos_ranks[best_is_param])

    pbo = np.mean(np.array(is_best_oos_ranks) <= 0.5)
    return pbo


def compute_walk_forward(ret_pivot, strat_type='b2', N_val=5, is_months=12, oos_months=3):
    """Compute Walk-Forward OOS Net Sharpe for specific strategy."""
    K_grid = [3, 5, 7, 10, 12, 15]
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    n_days = len(dates)

    is_days = is_months * 21
    oos_days = oos_months * 21

    param_daily_returns = {}
    for K_val in K_grid:
        if strat_type == 'a1':
            _, d_df = run_test_a1(ret_pivot, K=K_val, N=N_val, cost_bps=COST_BPS)
        else:
            _, d_df = run_test_b2(ret_pivot, K=K_val, N=N_val, cost_bps=COST_BPS)
        param_daily_returns[K_val] = d_df.set_index('date')['net_return']

    wf_oos_returns = []
    wf_dates = []

    start_idx = is_days
    while start_idx + oos_days <= n_days:
        is_start = start_idx - is_days
        is_end = start_idx
        oos_end = start_idx + oos_days

        is_date_range = dates[is_start:is_end]
        oos_date_range = dates[is_end:oos_end]

        best_K = None
        best_is_sharpe = -999.0

        for K_val in K_grid:
            s_is = param_daily_returns[K_val].reindex(is_date_range).fillna(0.0)
            sharpe_is = s_is.mean() / (s_is.std() + 1e-8) * np.sqrt(252)
            if sharpe_is > best_is_sharpe:
                best_is_sharpe = sharpe_is
                best_K = K_val

        s_oos = param_daily_returns[best_K].reindex(oos_date_range).fillna(0.0)
        wf_oos_returns.extend(s_oos.values)
        wf_dates.extend(oos_date_range)

        start_idx += oos_days

    wf_df = pd.DataFrame({'date': wf_dates, 'net_return': wf_oos_returns})
    wf_df['cum_net'] = np.exp(np.cumsum(wf_df['net_return']))

    years_oos = len(wf_df) / 252
    cum_net_oos = wf_df['cum_net'].iloc[-1] - 1.0
    cagr_oos = (1.0 + cum_net_oos) ** (1.0 / max(years_oos, 0.1)) - 1.0
    vol_oos = wf_df['net_return'].std() * np.sqrt(252)
    sharpe_oos = cagr_oos / vol_oos if vol_oos > 0 else np.nan

    return sharpe_oos, cum_net_oos


# ============================================================
# MONTHLY HEATMAP GENERATOR
# ============================================================
def generate_monthly_heatmap(daily_df, title_name, file_suffix):
    """Generate monthly return matrix and heatmap PNG."""
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_ret = daily_df.set_index('date')['net_return']

    monthly_ret = daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df = monthly_ret.unstack(level=1)

    month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                   'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    monthly_df.columns = [month_names[c - 1] for c in monthly_df.columns if c <= len(month_names)]

    yearly_ret = daily_ret.groupby(daily_ret.index.year).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df['Yıllık Toplam'] = yearly_ret

    monthly_df.to_csv(os.path.join(OUTPUT_DIR, f'21_monthly_returns_{file_suffix}.csv'))

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(monthly_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=True, ax=ax, linewidths=0.5)
    ax.set_title(f'{title_name} Aylık Net Getiri Matrisi (%)\n(2.0 BPS Komisyon, 2.0x VİOP Kaldıraçlı)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Yıl')
    ax.set_xlabel('Ay')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'21_heatmap_{file_suffix}.png'))
    plt.close()

    return monthly_df


# ============================================================
# MAIN COMPARATIVE EXECUTION
# ============================================================
def main():
    print("=" * 70)
    print("BIST 100 NİHAİ KARŞILAŞTIRMALI BACKTEST MOTORU (A1 1x1, B2 5x5, B2 1x1)")
    print("======================================================================")

    ret_pivot = load_data()

    strats = {
        'Test A1 (1x1)': ('a1', 10, 1, 'a1_1x1'),
        'Test B2 (5x5)': ('b2', 10, 5, 'b2_5x5'),
        'Test B2 (1x1)': ('b2', 10, 1, 'b2_1x1'),
    }

    metrics_list = []
    validation_list = []
    results_dict = {}
    monthly_matrices = {}

    for strat_name, (st_type, K_val, N_val, file_suf) in strats.items():
        print(f"\n---> Koşturuluyor: {strat_name}...")

        if st_type == 'a1':
            df_res, daily_df = run_test_a1(ret_pivot, K=K_val, N=N_val, cost_bps=COST_BPS)
        else:
            df_res, daily_df = run_test_b2(ret_pivot, K=K_val, N=N_val, cost_bps=COST_BPS)

        results_dict[strat_name] = df_res

        # 1. Metrics
        m = compute_metrics(strat_name, df_res, daily_df)
        metrics_list.append(m)

        # 2. Newey-West
        t_stat, p_val = compute_newey_west(daily_df, max_lag=10)

        # 3. PBO
        pbo = compute_pbo_cscv(ret_pivot, strat_type=st_type, N_val=N_val, S=16)

        # 4. Walk Forward OOS
        wf_sharpe, wf_cum = compute_walk_forward(ret_pivot, strat_type=st_type, N_val=N_val)

        validation_list.append({
            'Strateji': strat_name,
            'Newey-West t-stat': t_stat,
            'Newey-West p-value': p_val,
            'CSCV PBO (%)': pbo * 100.0,
            'Walk-Forward OOS Sharpe': wf_sharpe,
            'Walk-Forward OOS Total Return (%)': wf_cum * 100.0,
        })

        # 5. Monthly Heatmap
        monthly_matrices[strat_name] = generate_monthly_heatmap(daily_df, strat_name, file_suf)

    # Export Consolidated Summaries
    metrics_df = pd.DataFrame(metrics_list)
    validation_df = pd.DataFrame(validation_list)

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, '21_all3_metrics_summary.csv'), index=False)
    validation_df.to_csv(os.path.join(OUTPUT_DIR, '21_all3_validation_summary.csv'), index=False)

    # Master Plot: Comparative Equity Curves
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = {'Test A1 (1x1)': '#ff7f0e', 'Test B2 (5x5)': '#1f77b4', 'Test B2 (1x1)': '#2ca02c'}
    for name, df in results_dict.items():
        dt_idx = pd.to_datetime(df['datetime'])
        ax.plot(dt_idx, df['cum_net'], label=name, color=colors[name], linewidth=2.5)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}x'))
    ax.set_title('BIST 100 3 Şampiyon Strateji Karşılaştırmalı Net Getiri Eğrileri\n(Test A1 1x1 vs Test B2 5x5 vs Test B2 1x1, 2.0 BPS Komisyon, 2.0x VİOP Kaldıraçlı)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '21_all3_equity_curves.png'))
    plt.close()

    print("\n" + "=" * 70)
    print("3 ŞAMPİYON STRATEJİ KARŞILAŞTIRMALI METRİKLER TABLOSU")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("3 ŞAMPİYON STRATEJİ İSTATİSTİKSEL DOĞRULAMA TABLOSU")
    print("=" * 70)
    print(validation_df.to_string(index=False))


if __name__ == '__main__':
    main()
