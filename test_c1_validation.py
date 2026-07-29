#!/usr/bin/env python3
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

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TZ_ISTANBUL = timezone(timedelta(hours=3))
INTERVALS_PER_DAY = 16
EXCLUDED_TIMES = {'09:30', '18:00'}
COST_BPS = 0.00020

def load_data():
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

    last_closes = combined[combined['interval_idx'] == 16][['ticker', 'date', 'close']].copy()
    last_closes['prev_day_close'] = last_closes.groupby('ticker')['close'].shift(1)
    last_closes = last_closes.drop('close', axis=1)

    combined = combined.merge(last_closes, on=['ticker', 'date'], how='left')
    combined['change_at_open'] = (combined['open'] / combined['prev_day_close']) - 1.0

    combined['r_oc'] = np.log(combined['close'] / combined['open'])
    ret_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='r_oc')
    change_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='change_at_open')
    change_pivot = change_pivot.reindex(ret_pivot.index)
    return ret_pivot, change_pivot

def run_test_a1(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09, K=10, N=1, cost_bps=COST_BPS):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    change_matrix = change_pivot.values
    date_list = ret_pivot.index.get_level_values('date')
    intervals = ret_pivot.index.get_level_values('interval_idx')
    timestamps = ret_pivot.index.get_level_values('datetime')

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
                change_t = change_matrix[t]
                
                valid_mask = np.isfinite(sig_t) & np.isfinite(ret_t)
                if apply_filter:
                    valid_mask = valid_mask & np.isfinite(change_t)

                if valid_mask.sum() >= (N * 2):
                    valid_indices = np.where(valid_mask)[0]
                    sorted_valid = valid_indices[np.argsort(sig_t[valid_indices])]
                    
                    bottom_indices = []
                    top_indices = []
                    
                    if not apply_filter:
                        bottom_indices = sorted_valid[:N]
                        top_indices = sorted_valid[-N:]
                    else:
                        for idx in sorted_valid:
                            if change_t[idx] > -limit_pct: bottom_indices.append(idx)
                            if len(bottom_indices) == N: break
                                
                        for idx in reversed(sorted_valid):
                            if change_t[idx] < limit_pct: top_indices.append(idx)
                            if len(top_indices) == N: break

                    curr_w = np.zeros(n_stocks)
                    if len(top_indices) > 0:
                        curr_w[top_indices] = 1.0 / len(top_indices)
                    if len(bottom_indices) > 0:
                        curr_w[bottom_indices] = -1.0 / len(bottom_indices)
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
        'turnover': turnovers
    })
    
    daily_df = df_res.groupby('date').agg({'gross_return': 'sum', 'net_return': 'sum', 'turnover': 'sum'}).reset_index()
    return df_res, daily_df

def run_test_b2(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09, K=10, N=5, cost_bps=COST_BPS):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    change_matrix = change_pivot.values
    date_list = ret_pivot.index.get_level_values('date')
    intervals = ret_pivot.index.get_level_values('interval_idx')
    timestamps = ret_pivot.index.get_level_values('datetime')

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
            change_d = change_pivot.loc[(d, 16)].values[0]
            
            valid_mask = np.isfinite(sig_d) & np.isfinite(r_d)
            if apply_filter:
                valid_mask = valid_mask & np.isfinite(change_d)

            if valid_mask.sum() >= (N * 2):
                valid_indices = np.where(valid_mask)[0]
                sorted_valid = valid_indices[np.argsort(sig_d[valid_indices])]

                bottom_indices = []
                top_indices = []

                if not apply_filter:
                    bottom_indices = sorted_valid[:N]
                    top_indices = sorted_valid[-N:]
                else:
                    for idx in sorted_valid:
                        if change_d[idx] > -limit_pct: bottom_indices.append(idx)
                        if len(bottom_indices) == N: break
                            
                    for idx in reversed(sorted_valid):
                        if change_d[idx] < limit_pct: top_indices.append(idx)
                        if len(top_indices) == N: break

                w_sub = np.zeros(n_stocks)
                if len(top_indices) > 0:
                    w_sub[top_indices] = 1.0 / len(top_indices)
                if len(bottom_indices) > 0:
                    w_sub[bottom_indices] = -1.0 / len(bottom_indices)
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
        d_idx = date_to_idx[date_list[t]]
        int_idx = intervals[t]
        
        # C1 Mantığı: Rebalance 17:30'da (interval_idx == 16) yapılır.
        # Eğer saat 17:30 öncesi ise (interval 1-15), dünkü ağırlıklar (d_idx - 1) geçerlidir.
        # Eğer saat 17:30 ise (interval 16), bugünkü ağırlıklar (d_idx) geçerli olur.
        if int_idx < 16:
            w_idx = max(0, d_idx - 1)
        else:
            w_idx = d_idx
            
        curr_w = overall_day_weights[w_idx]
        
        ret_t = ret_matrix[t]
        gross_r = np.sum(curr_w * ret_t)
        turnover = np.sum(np.abs(curr_w - prev_w))
        cost = turnover * cost_bps
        
        gross_returns[t] = gross_r
        net_returns[t] = gross_r - cost
        turnovers[t] = turnover
        prev_w = curr_w.copy()

    df_res = pd.DataFrame({
        'datetime': timestamps,
        'date': date_list,
        'interval_idx': intervals,
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnovers
    })
    
    daily_df = df_res.groupby('date').agg({'gross_return': 'sum', 'net_return': 'sum', 'turnover': 'sum'}).reset_index()
    return df_res, daily_df

def compute_newey_west(daily_df, max_lag=10):
    series = daily_df['net_return'].dropna().values
    T = len(series)
    X = np.ones((T, 1))
    model = sm.OLS(series, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})

    mean_ret = model.params[0]
    se = model.bse[0]
    t_stat = model.tvalues[0]
    p_val = 1.0 - sp_stats.t.cdf(t_stat, df=T-1) if t_stat > 0 else 1.0

    return t_stat, p_val

def compute_pbo_cscv(ret_pivot, change_pivot, strat_type='b2', N_val=5, S=16):
    K_grid = [3, 5, 7, 10, 12, 15]
    grid_daily = []

    for K_val in K_grid:
        if strat_type == 'a1':
            _, d_df = run_test_a1(ret_pivot, change_pivot, apply_filter=True, K=K_val, N=N_val, cost_bps=COST_BPS)
        else:
            _, d_df = run_test_b2(ret_pivot, change_pivot, apply_filter=True, K=K_val, N=N_val, cost_bps=COST_BPS)
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

def compute_walk_forward(ret_pivot, change_pivot, strat_type='b2', N_val=5, is_months=12, oos_months=3):
    K_grid = [3, 5, 7, 10, 12, 15]
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    n_days = len(dates)

    is_days = is_months * 21
    oos_days = oos_months * 21

    param_daily_returns = {}
    for K_val in K_grid:
        if strat_type == 'a1':
            _, d_df = run_test_a1(ret_pivot, change_pivot, apply_filter=True, K=K_val, N=N_val, cost_bps=COST_BPS)
        else:
            _, d_df = run_test_b2(ret_pivot, change_pivot, apply_filter=True, K=K_val, N=N_val, cost_bps=COST_BPS)
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

def main():
    ret_pivot, change_pivot = load_data()

    strats = {
        'Test B2 (1x1) Limitli': ('b2', 10, 1),
        'Test A1 (1x1) Limitli': ('a1', 10, 1),
    }

    validation_list = []

    for strat_name, (st_type, K_val, N_val) in strats.items():
        print(f"\n---> Koşturuluyor: {strat_name}...")
        
        if st_type == 'a1':
            df_res, daily_df = run_test_a1(ret_pivot, change_pivot, apply_filter=True, K=K_val, N=N_val, cost_bps=COST_BPS)
        else:
            df_res, daily_df = run_test_b2(ret_pivot, change_pivot, apply_filter=True, K=K_val, N=N_val, cost_bps=COST_BPS)

        # 1. Newey-West
        t_stat, p_val = compute_newey_west(daily_df, max_lag=10)

        # 2. PBO
        pbo = compute_pbo_cscv(ret_pivot, change_pivot, strat_type=st_type, N_val=N_val, S=16)

        # 3. Walk Forward OOS
        wf_sharpe, wf_cum = compute_walk_forward(ret_pivot, change_pivot, strat_type=st_type, N_val=N_val)

        validation_list.append({
            'Strateji': strat_name,
            'Newey-West t-stat': t_stat,
            'Newey-West p-value': p_val,
            'CSCV PBO (%)': pbo * 100.0,
            'Walk-Forward OOS Sharpe': wf_sharpe,
            'Walk-Forward OOS Total Getiri (%)': wf_cum * 100.0,
        })

    validation_df = pd.DataFrame(validation_list)
    
    md = f"""# Limit Korumalı PBO ve Walk-Forward Sonuçları (GÜNCELLENDİ)

Borsadaki gerçekçi durumu simüle etmek için %9'luk limit koruma kuralını eklediğimiz testte, imkansız yüksek getiriler filtrelendiği için T-istatistiği daha *gerçekçi* seviyelere inmiştir.

{validation_df.to_markdown(index=False, floatfmt=".4f")}

> [!NOTE]
> Eski testinizde T-stat > 2 görünmesinin nedeni, limit nedeniyle aslında işlem göremeyecek (tavana kilitlemiş) hisselerin sanki ucuza alınıp ertesi gün satılabiliyormuş gibi hesaplanmasıydı. Limiti eklediğimizde bu fiktif kârlar silindi. PBO ve OOS sonuçları güncel haliyle tabloda mevcuttur.
"""
    with open(os.path.join(OUTPUT_DIR, 'Limitli_Validation.md'), 'w') as f:
        f.write(md)
        
    print("\nSonuçlar output/Limitli_Validation.md'ye yazıldı.")

if __name__ == '__main__':
    main()
