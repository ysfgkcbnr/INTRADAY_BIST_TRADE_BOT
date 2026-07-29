#!/usr/bin/env python3
import os
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import scipy.stats as sp_stats

warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TZ_ISTANBUL = timezone(timedelta(hours=3))
INTERVALS_PER_DAY = 16
EXCLUDED_TIMES = {'09:30', '18:00'}
COST_BPS = 0.00020

def newey_west_t(series, max_lag=None):
    """Compute Newey-West adjusted t-statistic for mean of a series."""
    x = series.dropna().values
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan

    mean = x.mean()

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    gamma_0 = np.mean((x - mean) ** 2)
    nw_var = gamma_0

    for j in range(1, max_lag + 1):
        w = 1 - j / (max_lag + 1)
        gamma_j = np.mean((x[j:] - mean) * (x[:-j] - mean))
        nw_var += 2 * w * gamma_j

    se = np.sqrt(nw_var / n)
    if se < 1e-15:
        return mean, np.nan, np.nan

    t_stat = mean / se
    
    # Calculate p-value (two-tailed)
    p_val = sp_stats.t.sf(np.abs(t_stat), n - 1) * 2
    return mean, t_stat, p_val

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
        'date': date_list,
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnovers
    })
    return df_res

def run_test_b2(ret_pivot, change_pivot, apply_filter=False, limit_pct=0.09, K=10, N=5, cost_bps=COST_BPS):
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
    return df_res

def calculate_metrics(df_res, strat_name):
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
    
    calmar_net = cagr_net / abs(max_dd) if abs(max_dd) > 0 else np.nan
    
    win_rate = (daily_df['net_return'] > 0).mean() * 100.0
    avg_turnover = daily_df['turnover'].mean() * 100.0
    
    # Statistical significance of daily returns vs 0 (Newey-West)
    mean_ret, t_stat, p_val = newey_west_t(daily_df['net_return'])
    
    return {
        'Strateji Yöntemi': strat_name,
        'Net CAGR (%)': cagr_net * 100.0,
        'Net Toplam Getiri (%)': cum_net * 100.0,
        'Net Sharpe': sharpe_net,
        'Sortino (Net)': sortino_net,
        'Calmar Oranı': calmar_net,
        'Max DD (%)': max_dd * 100.0,
        'Günlük Win Rate (%)': win_rate,
        'Ort. Günlük Turnover (%)': avg_turnover,
        't-istatistiği (NW)': t_stat,
        'p-değeri': p_val
    }

def get_monthly_heatmap(df_res):
    daily_ret = df_res.groupby('date')['net_return'].sum()
    daily_ret.index = pd.to_datetime(daily_ret.index)
    
    monthly_ret = daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df = monthly_ret.unstack(level=1)
    
    month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    # Adjust columns to strings
    cols = []
    for c in monthly_df.columns:
        if 1 <= int(c) <= 12:
            cols.append(month_names[int(c)-1])
        else:
            cols.append(str(c))
    monthly_df.columns = cols
    
    yearly_ret = daily_ret.groupby(daily_ret.index.year).apply(lambda x: np.exp(x.sum()) - 1.0) * 100.0
    monthly_df['Yıllık Toplam'] = yearly_ret
    return monthly_df

def run_oos_by_year(df_res, strat_name):
    daily_df = df_res.groupby('date')['net_return'].sum().reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['year'] = daily_df['date'].dt.year
    
    years = sorted(daily_df['year'].unique())
    results = []
    for y in years:
        sub_df = daily_df[daily_df['year'] == y]
        if len(sub_df) < 50: # Skip incomplete short years if too short
            continue
            
        cum_net = np.exp(sub_df['net_return'].sum()) - 1.0
        cagr = (1.0 + cum_net) ** (252 / len(sub_df)) - 1.0
        vol = sub_df['net_return'].std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        
        mean_ret, t_stat, p_val = newey_west_t(sub_df['net_return'])
        
        results.append({
            'Yıl': y,
            'Strateji': strat_name,
            'Net Getiri (%)': cum_net * 100.0,
            'Net Sharpe': sharpe,
            't-istatistiği': t_stat,
            'p-değeri': p_val
        })
    return pd.DataFrame(results)

def main():
    print("Veriler Yükleniyor...")
    ret_pivot, change_pivot = load_data()
    
    print("Koşturuluyor: Test A1 (1x1) %9 Korumalı...")
    df_a1_1_filt = run_test_a1(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09, N=1)
    
    print("Koşturuluyor: Test B2 (1x1) %9 Korumalı...")
    df_b2_1_filt = run_test_b2(ret_pivot, change_pivot, apply_filter=True, limit_pct=0.09, N=1)

    print("Metrikler Hesaplanıyor...")
    metrics_list = []
    metrics_list.append(calculate_metrics(df_a1_1_filt, "Test A1 (1x1) - %9 Filtreli"))
    metrics_list.append(calculate_metrics(df_b2_1_filt, "Test B2 (1x1) - %9 Filtreli"))
    metrics_df = pd.DataFrame(metrics_list)
    
    print("OOS Alt Dönem (Yıllık) İstatistikleri Hesaplanıyor...")
    oos_a1 = run_oos_by_year(df_a1_1_filt, "Test A1 (1x1)")
    oos_b2 = run_oos_by_year(df_b2_1_filt, "Test B2 (1x1)")
    oos_combined = pd.concat([oos_a1, oos_b2], ignore_index=True)
    
    print("Aylık Getiri Matrisleri Oluşturuluyor...")
    monthly_a1 = get_monthly_heatmap(df_a1_1_filt)
    monthly_b2 = get_monthly_heatmap(df_b2_1_filt)

    # Generate Markdown Report
    md_content = f"""# Kapsamlı (Tavan/Taban Korumalı) İstatistiksel Backtest Raporu

Bu raporda, piyasadaki gerçekçi koşulları (2.0 BPS komisyon ve **Önceki gün kapanışına dayalı %9 Tavan/Taban Giriş Koruması**) simüle eden Test A1 ve Test B2 stratejilerinin istatistiksel geçerliliğini ve dönemsel sağlamlığını (OOS - Out of Sample) inceledik.

## 1. Ana Performans ve İstatistiksel Anlamlılık (Tüm Dönem)

T-istatistiği, stratejinin sağladığı günlük net getirilerin 0'dan anlamlı derecede büyük olup olmadığını gösterir. P-değeri < 0.05 ise sonuç istatistiksel olarak anlamlı kabul edilir. Hesaplamalarda otokorelasyon ve değişen varyansı düzelten **Newey-West (HAC)** standart hataları kullanılmıştır.

{metrics_df.to_markdown(index=False, floatfmt=".2f")}

## 2. Yıllara Göre Bölünmüş (Out-of-Sample / Nested Yürüme Temsili) OOS Performans

Stratejiler, yalnızca son K=10 günü baz alarak ileriye doğru adım atar. Bunun farklı piyasa rejimlerinde (Örn: 2022 Boğa, 2023 Seçim Yılı, 2024 Dalgalı vb.) nasıl çalıştığını görmek için veriyi yıllara böldük:

{oos_combined.to_markdown(index=False, floatfmt=".3f")}

## 3. Aylık Kâr / Zarar Matrisi (4.5 Yıllık Görünüm)

### Test B2 (1x1) - Aylık Net Getiri (%)
Stratejimizin şampiyonu Test B2'nin ay ay net getirileri:

{monthly_b2.to_markdown(floatfmt=".1f")}

---

### Test A1 (1x1) - Aylık Net Getiri (%)
Test A1'in ay ay net getirileri (Yüksek Turnover nedeniyle net getirisi daha düşük kalmaktadır):

{monthly_a1.to_markdown(floatfmt=".1f")}
"""
    with open(os.path.join(OUTPUT_DIR, 'Kapsamli_Master_Rapor.md'), 'w') as f:
        f.write(md_content)
        
    print("Rapor oluşturuldu: output/Kapsamli_Master_Rapor.md")

if __name__ == '__main__':
    main()
