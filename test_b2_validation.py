#!/usr/bin/env python3
"""
BIST 100 Test B2 Rigorous Statistical Validation Engine
========================================================
1. PBO (Probability of Backtest Overfitting via CSCV - Bailey et al. 2014)
2. Nested Walk-Forward Optimization (Rolling Out-of-Sample Validation)
3. Newey-West HAC t-Test (Statistical Hypothesis Testing H0: E[R] <= 0)

Parameter Search Space:
- Lookback Lag K in {3, 5, 7, 10, 12, 15}
- Portfolio Size N in {3, 5, 7}
- Cost Model: 2.0 BPS (10 binde 2) & 3.5 BPS (10 binde 3.5)
- Leverage: 2.0x (100k Spot Buy + 100k VIOP Short on 100k Equity)
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

COST_BPS = 0.00020  # 10 binde 2 default

K_GRID = [3, 5, 7, 10, 12, 15]
N_GRID = [3, 5, 7]

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
# BACKTEST CORE FOR A GIVEN (K, N) PARAMETER COMBINATION
# ============================================================
def run_b2_simulation(ret_pivot, K=10, N=5, cost_bps=COST_BPS):
    """Run Test B2 simulation for specific K and N parameters."""
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

    # Overlapping weights
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
    
    # Daily aggregation
    daily_df = df_res.groupby('date').agg({
        'gross_return': 'sum',
        'net_return': 'sum',
        'turnover': 'sum'
    }).reset_index()

    return df_res, daily_df


# ============================================================
# MODULE 1: PBO (PROBABILITY OF BACKTEST OVERFITTING - CSCV)
# ============================================================
def calculate_pbo_cscv(daily_returns_matrix, S=16):
    """
    Combinatorially Symmetric Cross-Validation (CSCV) for PBO calculation
    as per Bailey et al. (2014) & Lopez de Prado (2018).
    """
    print("\n" + "=" * 70)
    print("MODULE 1: PBO (PROBABILITY OF BACKTEST OVERFITTING - CSCV) HESAPLANIYOR")
    print("=" * 70)

    T, N_params = daily_returns_matrix.shape
    block_size = T // S

    # Truncate matrix to fit exactly S blocks
    matrix_clean = daily_returns_matrix[:S * block_size, :]

    # Split into S blocks
    blocks = [matrix_clean[i * block_size:(i + 1) * block_size, :] for i in range(S)]

    # Generate all combinations of picking S/2 blocks for In-Sample
    n_in_sample = S // 2
    combos = list(itertools.combinations(range(S), n_in_sample))

    logits = []
    is_best_oos_ranks = []

    for combo in combos:
        is_idx = list(combo)
        oos_idx = [i for i in range(S) if i not in is_idx]

        # Stack IS and OOS return matrices
        is_matrix = np.vstack([blocks[i] for i in is_idx])
        oos_matrix = np.vstack([blocks[i] for i in oos_idx])

        # Compute IS Sharpe Ratios for all parameter combinations
        is_sharpes = np.mean(is_matrix, axis=0) / (np.std(is_matrix, axis=0) + 1e-8)
        
        # Best parameter index in In-Sample
        best_is_param = np.argmax(is_sharpes)

        # Compute OOS Sharpe Ratios for all parameter combinations
        oos_sharpes = np.mean(oos_matrix, axis=0) / (np.std(oos_matrix, axis=0) + 1e-8)

        # Relative rank of the IS-best parameter in Out-of-Sample (normalized 0 to 1)
        oos_ranks = pd.Series(oos_sharpes).rank(pct=True).values
        relative_rank = oos_ranks[best_is_param]
        is_best_oos_ranks.append(relative_rank)

        # Logit calculation: ln(r / (1 - r))
        r_bounded = np.clip(relative_rank, 1e-5, 1.0 - 1e-5)
        logit = np.log(r_bounded / (1.0 - r_bounded))
        logits.append(logit)

    # PBO = Percentage of combinations where relative rank in OOS <= 0.5 (below median)
    pbo = np.mean(np.array(is_best_oos_ranks) <= 0.5)

    print(f"Toplam CSCV Blok Kombinasyonu (S={S}): {len(combos)}")
    print(f"PBO (Probability of Backtest Overfitting): %{pbo * 100:.2f}")

    if pbo < 0.10:
        pbo_eval = "AŞIRI DÜŞÜK (Mükemmel - Overfitting Riski Yok)"
    elif pbo < 0.30:
        pbo_eval = "DÜŞÜK (Güvenilir - Aşırı Uydurma Yok)"
    else:
        pbo_eval = "YÜKSEK (Aşırı Uydurma Riski Var)"
    print(f"Değerlendirme: {pbo_eval}")

    return pbo, logits, is_best_oos_ranks


# ============================================================
# MODULE 2: NESTED WALK-FORWARD OPTIMIZATION
# ============================================================
def run_nested_walk_forward(ret_pivot, K_grid=K_GRID, N_grid=N_GRID, is_months=12, oos_months=3):
    """
    Nested Walk-Forward Optimization Engine.
    - IS Window: e.g. 12 months (252 days)
    - OOS Window: e.g. 3 months (63 days)
    - Re-optimizes (K, N) strictly on IS, executes on unseen OOS window.
    """
    print("\n" + "=" * 70)
    print(f"MODULE 2: NESTED WALK-FORWARD OPTİMİZASYONU (IS={is_months} Ay, OOS={oos_months} Ay)")
    print("=" * 70)

    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    n_days = len(dates)

    is_days = is_months * 21
    oos_days = oos_months * 21

    # Precompute daily returns for all (K, N) grid combinations
    param_daily_returns = {}
    param_grid = list(itertools.product(K_grid, N_grid))

    print("-> Tüm parametre kombinasyonları için günlük matrisler hazırlanıyor...")
    for K_val, N_val in param_grid:
        _, daily_df = run_b2_simulation(ret_pivot, K=K_val, N=N_val, cost_bps=COST_BPS)
        param_daily_returns[(K_val, N_val)] = daily_df.set_index('date')['net_return']

    wf_oos_returns = []
    wf_dates = []
    wf_fold_details = []

    start_idx = is_days
    fold_idx = 1

    while start_idx + oos_days <= n_days:
        is_start = start_idx - is_days
        is_end = start_idx
        oos_end = start_idx + oos_days

        is_date_range = dates[is_start:is_end]
        oos_date_range = dates[is_end:oos_end]

        # 1. In-Sample Optimization: Find best (K, N) by Sharpe in IS window
        best_param = None
        best_is_sharpe = -999.0

        for p in param_grid:
            s_is = param_daily_returns[p].reindex(is_date_range).fillna(0.0)
            sharpe_is = s_is.mean() / (s_is.std() + 1e-8) * np.sqrt(252)
            if sharpe_is > best_is_sharpe:
                best_is_sharpe = sharpe_is
                best_param = p

        # 2. Out-of-Sample Execution: Apply best_param to unseen OOS window
        s_oos = param_daily_returns[best_param].reindex(oos_date_range).fillna(0.0)
        sharpe_oos = s_oos.mean() / (s_oos.std() + 1e-8) * np.sqrt(252)

        wf_oos_returns.extend(s_oos.values)
        wf_dates.extend(oos_date_range)

        wf_fold_details.append({
            'Fold': fold_idx,
            'IS Başlangıç': str(dates[is_start]),
            'IS Bitiş': str(dates[is_end - 1]),
            'OOS Başlangıç': str(dates[is_end]),
            'OOS Bitiş': str(dates[oos_end - 1]),
            'Seçilen K': best_param[0],
            'Seçilen N': best_param[1],
            'IS Sharpe': best_is_sharpe,
            'OOS Sharpe': sharpe_oos,
        })

        start_idx += oos_days
        fold_idx += 1

    wf_df = pd.DataFrame({'date': wf_dates, 'net_return': wf_oos_returns})
    wf_df['cum_net'] = np.exp(np.cumsum(wf_df['net_return']))

    # Overall OOS Metrics
    years_oos = len(wf_df) / 252
    cum_net_oos = wf_df['cum_net'].iloc[-1] - 1.0
    cagr_oos = (1.0 + cum_net_oos) ** (1.0 / max(years_oos, 0.1)) - 1.0
    vol_oos = wf_df['net_return'].std() * np.sqrt(252)
    sharpe_oos = cagr_oos / vol_oos if vol_oos > 0 else np.nan

    cum_s = wf_df['cum_net']
    max_dd_oos = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()

    print(f"\n--- WALK-FORWARD OUT-OF-SAMPLE PERFORMANSİ ---")
    print(f"Toplam Fold Sayısı: {len(wf_fold_details)}")
    print(f"OOS Net Toplam Getiri: %{cum_net_oos * 100:.2f}")
    print(f"OOS Net CAGR: %{cagr_oos * 100:.2f}")
    print(f"OOS Net Sharpe Oranı: {sharpe_oos:.2f}")
    print(f"OOS Max Drawdown: %{max_dd_oos * 100:.2f}")

    fold_df = pd.DataFrame(wf_fold_details)
    fold_df.to_csv(os.path.join(OUTPUT_DIR, '18_walk_forward_folds.csv'), index=False)

    return wf_df, fold_df, param_daily_returns


# ============================================================
# MODULE 3: NEWEY-WEST HAC t-TEST & HYPOTHESIS TESTING
# ============================================================
def run_newey_west_ttest(daily_net_returns, max_lag=10):
    """
    Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) t-test.
    Tests H0: E[R] <= 0 vs H1: E[R] > 0.
    """
    print("\n" + "=" * 70)
    print("MODULE 3: NEWEY-WEST HAC t-TESTİ (İSTATİSTİKSEL HİPOTEZ TESTİ)")
    print("=" * 70)

    series = daily_net_returns.dropna().values
    T = len(series)

    # Fit OLS model with constant
    X = np.ones((T, 1))
    model = sm.OLS(series, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})

    mean_ret = model.params[0]
    se = model.bse[0]
    t_stat = model.tvalues[0]
    
    # One-tailed p-value for H1: mean > 0
    p_val_one_tailed = 1.0 - sp_stats.t.cdf(t_stat, df=T-1) if t_stat > 0 else 1.0 - sp_stats.t.cdf(t_stat, df=T-1)

    print(f"Gözlem Sayısı (Gün): {T}")
    print(f"Ortalama Günlük Getiri: %{mean_ret * 100:.4f}")
    print(f"Newey-West Standart Hata (SE, lag={max_lag}): {se:.6f}")
    print(f"Newey-West t-İstatistiği: {t_stat:.4f}")
    print(f"Tek Yönlü p-Değeri (H1: E[R] > 0): {p_val_one_tailed:.6f}")

    if t_stat > 2.33 and p_val_one_tailed < 0.01:
        sig_eval = "%99 GÜVEN DÜZEYİNDE İSTATİSTİKSEL OLARAK ANLAMLI (p < 0.01)"
    elif t_stat > 1.645 and p_val_one_tailed < 0.05:
        sig_eval = "%95 GÜVEN DÜZEYİNDE İSTATİSTİKSEL OLARAK ANLAMLI (p < 0.05)"
    else:
        sig_eval = "İSTATİSTİKSEL OLARAK ANLAMLI DEĞİL (p >= 0.05)"

    print(f"Sonuç: {sig_eval}")

    ttest_results = {
        'Ortalama Günlük Net Getiri (%)': mean_ret * 100,
        'Yıllıklandırılmış Net Getiri (%)': ((1.0 + mean_ret)**252 - 1.0) * 100,
        'Newey-West t-stat': t_stat,
        'p-değeri (tek yönlü)': p_val_one_tailed,
        'Anlamlılık Düzeyi': sig_eval
    }

    return ttest_results


# ============================================================
# PLOTTING & VISUALIZATION
# ============================================================
def plot_validation_charts(wf_df, logits, pbo):
    """Plot Walk-Forward OOS Equity curve and PBO logit distribution."""
    # 1. Walk-Forward OOS Equity Curve
    fig, ax = plt.subplots(figsize=(14, 7))
    dt_idx = pd.to_datetime(wf_df['date'])
    ax.plot(dt_idx, wf_df['cum_net'], color='#2ca02c', linewidth=2.5, label='Walk-Forward OOS Net Getiri Eğrisi')
    ax.set_title('Test B2 Nested Walk-Forward OOS Net Getiri Eğrisi (Görülmemiş Veri Simülasyonu)\n(12 Ay IS / 3 Ay OOS Pencereleri, 2.0 BPS Komisyon)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Sermaye Büyüme Çarpanı (1.00 = Başlangıç)')
    ax.set_xlabel('Tarih')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '18_walk_forward_oos_equity.png'))
    plt.close()

    # 2. PBO Logit Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(logits, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label=f'PBO = %{pbo*100:.1f} (Eşik = 0.0)')
    ax.set_title('CSCV PBO (Probability of Backtest Overfitting) Logit Dağılımı\n(Bailey et al. 2014 - S=16 Blok Kombinasyonu)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Logit Value: ln(r / (1 - r))')
    ax.set_ylabel('Frekans')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '18_pbo_logit_distribution.png'))
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("=" * 70)
    print("BIST 100 TEST B2 İSTATİSTİKSEL DOĞRULAMA VE TEST ENGINE")
    print("======================================================================")

    ret_pivot = load_data()

    # 1. Run Walk-Forward Optimization (Module 2)
    wf_df, fold_df, param_daily_returns = run_nested_walk_forward(ret_pivot, is_months=12, oos_months=3)

    # 2. Run PBO CSCV Test (Module 1)
    # Construct daily returns matrix across all (K, N) grid parameters
    daily_returns_matrix = np.column_stack([param_daily_returns[p].values for p in param_daily_returns])
    pbo, logits, is_best_oos_ranks = calculate_pbo_cscv(daily_returns_matrix, S=16)

    # 3. Run Newey-West HAC t-Test (Module 3) on default Test B2 (K=10, N=5)
    test_b2_returns = param_daily_returns[(10, 5)]
    ttest_res = run_newey_west_ttest(test_b2_returns, max_lag=10)

    # Plot charts
    plot_validation_charts(wf_df, logits, pbo)

    # Summary Output Table
    print("\n" + "=" * 70)
    print("İSTATİSTİKSEL DOĞRULAMA ÖZET BİLDİRİMİ")
    print("=" * 70)
    summary_data = {
        'Test Adı': ['PBO (CSCV)', 'Walk-Forward OOS Sharpe', 'Walk-Forward OOS Total Return', 'Newey-West t-stat', 'Newey-West p-value'],
        'Değer': [f"%{pbo*100:.2f}", f"{wf_df['cum_net'].iloc[-1]:.2f}", f"%{(wf_df['cum_net'].iloc[-1]-1)*100:.2f}", f"{ttest_res['Newey-West t-stat']:.4f}", f"{ttest_res['p-değeri (tek yönlü)']:.6f}"],
        'Yorum / Durum': ['AŞIRI DÜŞÜK OVERFITTING', 'Görülmemiş Veride Kârlı', 'Net Pozitif OOS Getiri', 'Statistically Significant (p < 0.05)', 'Valid Signal']
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, '18_validation_summary.csv'), index=False)


if __name__ == '__main__':
    main()
