#!/usr/bin/env python3
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
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

    # Doğru Tavan/Taban hesaplaması için önceki günün kapanışını bul
    # Her günün son barı (interval 16) kapanışını alıyoruz
    last_closes = combined[combined['interval_idx'] == 16][['ticker', 'date', 'close']].copy()
    last_closes['prev_day_close'] = last_closes.groupby('ticker')['close'].shift(1)
    last_closes = last_closes.drop('close', axis=1)

    combined = combined.merge(last_closes, on=['ticker', 'date'], how='left')
    
    # Giriş anındaki (barın açılışındaki) limit yüzdesi
    combined['change_at_open'] = (combined['open'] / combined['prev_day_close']) - 1.0

    combined['r_oc'] = np.log(combined['close'] / combined['open'])
    ret_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='r_oc')
    change_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='change_at_open')
    change_pivot = change_pivot.reindex(ret_pivot.index)
    return ret_pivot, change_pivot

def run_test_a1(ret_pivot, change_pivot, apply_filter=False, limit_pct=0.09, K=10, N=1, cost_bps=COST_BPS):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    change_matrix = change_pivot.values
    date_list = ret_pivot.index.get_level_values('date')
    intervals = ret_pivot.index.get_level_values('interval_idx')

    gross_returns = np.zeros(n_rows)
    net_returns = np.zeros(n_rows)
    turnovers = np.zeros(n_rows)
    prev_w = np.zeros(n_stocks)
    blocked_longs = 0
    blocked_shorts = 0

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
                            else: blocked_shorts += 1
                            if len(bottom_indices) == N: break
                                
                        for idx in reversed(sorted_valid):
                            if change_t[idx] < limit_pct: top_indices.append(idx)
                            else: blocked_longs += 1
                            if len(top_indices) == N: break

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
        'date': date_list,
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnovers
    })
    return df_res, blocked_longs, blocked_shorts

def run_test_b2(ret_pivot, change_pivot, apply_filter=False, limit_pct=0.09, K=10, N=5, cost_bps=COST_BPS):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    change_matrix = change_pivot.values
    date_list = ret_pivot.index.get_level_values('date')

    n_days = len(dates)
    sub_weights = np.zeros((n_days, n_stocks))
    blocked_longs = 0
    blocked_shorts = 0

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
                        else: blocked_shorts += 1
                        if len(bottom_indices) == N: break
                            
                    for idx in reversed(sorted_valid):
                        if change_d[idx] < limit_pct: top_indices.append(idx)
                        else: blocked_longs += 1
                        if len(top_indices) == N: break

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
        d_idx = date_to_idx[date_list[t]]
        curr_w = overall_day_weights[d_idx]
        
        ret_t = ret_matrix[t]
        gross_r = np.sum(curr_w * ret_t)
        turnover = np.sum(np.abs(curr_w - prev_w))
        cost = turnover * cost_bps
        
        gross_returns[t] = gross_r
        net_returns[t] = gross_r - cost
        turnovers[t] = turnover
        prev_w = curr_w

    df_res = pd.DataFrame({
        'date': date_list,
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnovers
    })
    return df_res, blocked_longs, blocked_shorts

def calculate_metrics(df_res, strat_name, bl, bs):
    daily_df = df_res.groupby('date').sum().reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    n_days = len(daily_df)
    years = n_days / 252
    
    cum_gross = np.exp(daily_df['gross_return'].sum()) - 1.0
    cum_net = np.exp(daily_df['net_return'].sum()) - 1.0
    cagr_gross = (1.0 + cum_gross) ** (1.0 / max(years, 0.1)) - 1.0
    cagr_net = (1.0 + cum_net) ** (1.0 / max(years, 0.1)) - 1.0
    
    vol_net = daily_df['net_return'].std() * np.sqrt(252)
    sharpe_net = cagr_net / vol_net if vol_net > 0 else np.nan
    
    downside_net = daily_df[daily_df['net_return'] < 0]['net_return']
    downside_vol = downside_net.std() * np.sqrt(252) if len(downside_net) > 0 else np.nan
    sortino_net = cagr_net / downside_vol if downside_vol > 0 else np.nan
    
    cum_series = np.exp(np.cumsum(daily_df['net_return']))
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    max_dd = dd.min()
    
    win_rate = (daily_df['net_return'] > 0).mean() * 100.0
    avg_turnover = daily_df['turnover'].mean() * 100.0
    
    return {
        'Strateji Yöntemi': strat_name,
        'Net CAGR (%)': round(cagr_net * 100.0, 2),
        'Net Toplam Getiri (%)': round(cum_net * 100.0, 2),
        'Net Sharpe': round(sharpe_net, 2),
        'Sortino (Net)': round(sortino_net, 2),
        'Max DD (%)': round(max_dd * 100.0, 2),
        'Günlük Win Rate (%)': round(win_rate, 2),
        'Ort. Günlük Turnover (%)': round(avg_turnover, 2),
        'Engellenen Long': bl,
        'Engellenen Short': bs
    }

def main():
    print("Veriler Yükleniyor...")
    ret_pivot, change_pivot = load_data()
    
    results = []
    
    print("Koşturuluyor: Test A1 (1x1) Orijinal...")
    df_a1_1_orig, bl, bs = run_test_a1(ret_pivot, change_pivot, apply_filter=False, N=1)
    results.append(calculate_metrics(df_a1_1_orig, "Test A1 (1x1) - Orijinal", bl, bs))
    
    print("Koşturuluyor: Test A1 (1x1) %9 Korumalı...")
    df_a1_1_filt, bl, bs = run_test_a1(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09, N=1)
    results.append(calculate_metrics(df_a1_1_filt, "Test A1 (1x1) - %9 Filtreli", bl, bs))
    
    print("Koşturuluyor: Test B2 (1x1) Orijinal...")
    df_b2_1_orig, bl, bs = run_test_b2(ret_pivot, change_pivot, apply_filter=False, N=1)
    results.append(calculate_metrics(df_b2_1_orig, "Test B2 (1x1) - Orijinal", bl, bs))
    
    print("Koşturuluyor: Test B2 (1x1) %9 Korumalı...")
    df_b2_1_filt, bl, bs = run_test_b2(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09, N=1)
    results.append(calculate_metrics(df_b2_1_filt, "Test B2 (1x1) - %9 Filtreli", bl, bs))

    print("Koşturuluyor: Test B2 (5x5) Orijinal...")
    df_b2_5_orig, bl, bs = run_test_b2(ret_pivot, change_pivot, apply_filter=False, N=5)
    results.append(calculate_metrics(df_b2_5_orig, "Test B2 (5x5) - Orijinal", bl, bs))
    
    print("Koşturuluyor: Test B2 (5x5) %9 Korumalı...")
    df_b2_5_filt, bl, bs = run_test_b2(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09, N=5)
    results.append(calculate_metrics(df_b2_5_filt, "Test B2 (5x5) - %9 Filtreli", bl, bs))
    
    with open('limit_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Testler tamamlandı, sonuçlar limit_test_results.json'a kaydedildi.")

if __name__ == '__main__':
    main()
