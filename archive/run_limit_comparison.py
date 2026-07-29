#!/usr/bin/env python3
import os
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TZ_ISTANBUL = timezone(timedelta(hours=3))
INTERVALS_PER_DAY = 16
EXCLUDED_TIMES = {'09:30', '18:00'}
N_TOP_BOTTOM = 5
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
    combined = combined.sort_values(['ticker', 'date', 'time_str'])
    
    combined['prev_day_close'] = combined.groupby('ticker')['close'].shift(INTERVALS_PER_DAY)
    combined['daily_change'] = (combined['close'] / combined['prev_day_close']) - 1.0

    combined = combined.sort_values(['date', 'time_str', 'ticker'])
    time_to_idx = {t: i + 1 for i, t in enumerate(sorted(combined['time_str'].unique()))}
    combined['interval_idx'] = combined['time_str'].map(time_to_idx)

    combined['r_oc'] = np.log(combined['close'] / combined['open'])
    ret_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='r_oc')
    change_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='daily_change')
    change_pivot = change_pivot.reindex(ret_pivot.index)
    return ret_pivot, change_pivot


def run_test_a1(ret_pivot, change_pivot, apply_filter=False, limit_pct=0.09, K=10):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    change_matrix = change_pivot.values
    date_list = ret_pivot.index.get_level_values('date')
    intervals = ret_pivot.index.get_level_values('interval_idx')

    net_returns = np.zeros(n_rows)
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

                if valid_mask.sum() >= (N_TOP_BOTTOM * 2):
                    valid_indices = np.where(valid_mask)[0]
                    sorted_valid = valid_indices[np.argsort(sig_t[valid_indices])]
                    
                    bottom_indices = []
                    top_indices = []
                    
                    if not apply_filter:
                        bottom_indices = sorted_valid[:N_TOP_BOTTOM]
                        top_indices = sorted_valid[-N_TOP_BOTTOM:]
                    else:
                        for idx in sorted_valid:
                            if change_t[idx] > -limit_pct: bottom_indices.append(idx)
                            if len(bottom_indices) == N_TOP_BOTTOM: break
                                
                        for idx in reversed(sorted_valid):
                            if change_t[idx] < limit_pct: top_indices.append(idx)
                            if len(top_indices) == N_TOP_BOTTOM: break

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
        
        net_returns[t] = gross_r - (turnover * COST_BPS)
        prev_w = curr_w

    return net_returns, date_list

def run_test_b2(ret_pivot, change_pivot, apply_filter=False, limit_pct=0.09, K=10):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    change_matrix = change_pivot.values
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
            change_d = change_pivot.loc[(d, 16)].values[0]
            
            valid_mask = np.isfinite(sig_d) & np.isfinite(r_d)
            if apply_filter:
                valid_mask = valid_mask & np.isfinite(change_d)

            if valid_mask.sum() >= (N_TOP_BOTTOM * 2):
                valid_indices = np.where(valid_mask)[0]
                sorted_valid = valid_indices[np.argsort(sig_d[valid_indices])]

                bottom_indices = []
                top_indices = []

                if not apply_filter:
                    bottom_indices = sorted_valid[:N_TOP_BOTTOM]
                    top_indices = sorted_valid[-N_TOP_BOTTOM:]
                else:
                    for idx in sorted_valid:
                        if change_d[idx] > -limit_pct: bottom_indices.append(idx)
                        if len(bottom_indices) == N_TOP_BOTTOM: break
                            
                    for idx in reversed(sorted_valid):
                        if change_d[idx] < limit_pct: top_indices.append(idx)
                        if len(top_indices) == N_TOP_BOTTOM: break

                w_sub = np.zeros(n_stocks)
                w_sub[top_indices] = 1.0 / N_TOP_BOTTOM
                w_sub[bottom_indices] = -1.0 / N_TOP_BOTTOM
                sub_weights[d_idx] = w_sub

    overall_day_weights = np.zeros((n_days, n_stocks))
    for d_idx in range(n_days):
        start_d = max(0, d_idx - K + 1)
        overall_day_weights[d_idx] = np.mean(sub_weights[start_d:d_idx + 1], axis=0)

    net_returns = np.zeros(n_rows)
    prev_w = np.zeros(n_stocks)

    for t in range(n_rows):
        d_idx = date_to_idx[date_list[t]]
        curr_w = overall_day_weights[d_idx]
        
        ret_t = ret_matrix[t]
        gross_r = np.sum(curr_w * ret_t)
        turnover = np.sum(np.abs(curr_w - prev_w))
        net_returns[t] = gross_r - (turnover * COST_BPS)
        prev_w = curr_w

    return net_returns, date_list


def calculate_metrics(net_returns, date_list, name):
    intervals_per_year = 252 * INTERVALS_PER_DAY
    df = pd.DataFrame({'net_return': net_returns, 'date': date_list})
    valid_df = df[df['net_return'] != 0.0]
    if len(valid_df) == 0: valid_df = df
    
    n_periods = len(valid_df)
    years = n_periods / intervals_per_year
    cum_net = np.exp(valid_df['net_return'].sum()) - 1.0
    cagr_net = (1.0 + cum_net) ** (1.0 / max(years, 0.1)) - 1.0
    
    vol_net = valid_df['net_return'].std() * np.sqrt(intervals_per_year)
    sharpe_net = cagr_net / vol_net if vol_net > 0 else np.nan
    
    downside_net = valid_df[valid_df['net_return'] < 0]['net_return']
    downside_vol = downside_net.std() * np.sqrt(intervals_per_year) if len(downside_net) > 0 else np.nan
    sortino_net = cagr_net / downside_vol if downside_vol > 0 else np.nan
    
    cum_series = np.exp(np.cumsum(valid_df['net_return']))
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    max_dd = dd.min()
    calmar = cagr_net / abs(max_dd) if abs(max_dd) > 0 else np.nan
    
    daily_returns = valid_df.groupby('date')['net_return'].sum()
    daily_win_rate = (daily_returns > 0).mean() * 100.0

    return {
        'Strateji': name,
        'CAGR (%)': cagr_net * 100.0,
        'Net Sharpe': sharpe_net,
        'Net Sortino': sortino_net,
        'Max DD (%)': max_dd * 100.0,
        'Calmar': calmar,
        'Gunluk Kazanma (%)': daily_win_rate
    }

def main():
    print("Loading data...")
    ret_pivot, change_pivot = load_data()
    
    print("Running Tests...")
    a1_net, a1_dates = run_test_a1(ret_pivot, change_pivot, apply_filter=False)
    a1f_net, a1f_dates = run_test_a1(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09)
    b2_net, b2_dates = run_test_b2(ret_pivot, change_pivot, apply_filter=False)
    b2f_net, b2f_dates = run_test_b2(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09)
    
    res = []
    res.append(calculate_metrics(a1_net, a1_dates, "A1 Orijinal"))
    res.append(calculate_metrics(a1f_net, a1f_dates, "A1 Filtreli"))
    res.append(calculate_metrics(b2_net, b2_dates, "B2 Orijinal"))
    res.append(calculate_metrics(b2f_net, b2f_dates, "B2 Filtreli"))
    
    df_res = pd.DataFrame(res)
    print("\n--- RISKS & METRICS ---")
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    main()
