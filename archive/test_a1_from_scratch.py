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
COST_BPS = 0.00020  # 2.0 BPS

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

    combined['r_oc'] = np.log(combined['close'] / combined['open'])
    ret_pivot = combined.pivot_table(index=['date', 'interval_idx', 'datetime'],
                                     columns='ticker', values='r_oc')
    return ret_pivot

def run_test_a1(ret_pivot, K=10, N=1, cost_bps=COST_BPS):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    n_rows, n_stocks = ret_pivot.shape
    ret_matrix = ret_pivot.values
    date_list = ret_pivot.index.get_level_values('date')
    intervals = ret_pivot.index.get_level_values('interval_idx')

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
        'date': date_list,
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnovers
    })
    return df_res

def main():
    print("Veriler Yükleniyor...")
    ret_pivot = load_data()
    print("Test A1 (1x1) Çalıştırılıyor (0 BPS ve 2.0 BPS)...")
    df_res = run_test_a1(ret_pivot, N=1, cost_bps=COST_BPS)
    
    # Brüt ve Net CAGR Hesaplama
    daily_df = df_res.groupby('date').sum().reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    n_days = len(daily_df)
    years = n_days / 252
    
    cum_gross = np.exp(daily_df['gross_return'].sum()) - 1.0
    cum_net = np.exp(daily_df['net_return'].sum()) - 1.0
    
    cagr_gross = (1.0 + cum_gross) ** (1.0 / years) - 1.0
    cagr_net = (1.0 + cum_net) ** (1.0 / years) - 1.0
    
    avg_turnover = daily_df['turnover'].mean() * 100.0
    
    print("\n==================================================")
    print("A1 (1x1) STRATEJİSİ - SIFIRDAN BACKTEST SONUÇLARI")
    print("==================================================")
    print(f"Brüt CAGR (Sıfır Komisyon) : {cagr_gross*100:.2f}% (Gerçek dışı senaryo)")
    print(f"Net CAGR (2.0 BPS Komisyon): {cagr_net*100:.2f}% (GERÇEK DÜNYA)")
    print(f"Günlük Ortalama İşlem Hacmi (Turnover): {avg_turnover:.2f}%")
    print("==================================================\n")
    
    # Aylık Net Getiriler
    daily_ret = daily_df.set_index('date')['net_return']
    monthly_ret = daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df = monthly_ret.unstack(level=1).round(2)
    monthly_df.columns = [f"Ay-{c}" for c in monthly_df.columns]
    
    yearly_ret = daily_ret.groupby(daily_ret.index.year).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df['Yillik Toplam'] = yearly_ret.round(2)
    
    print("AYLIK NET GETİRİLER TABLOSU (%) (Komisyon Düşüldükten Sonra):")
    print(monthly_df.to_string())

if __name__ == '__main__':
    main()
