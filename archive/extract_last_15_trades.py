#!/usr/bin/env python3
"""
Son 15 İşlem Detayını Çıkarma Scripti
3 Strateji için:
1. A1 (1x1)
2. B2 (5x5)
3. B2 (1x1)
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import timezone, timedelta

# ============================================================
# AYARLAR
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TZ_ISTANBUL = timezone(timedelta(hours=3))
INTERVALS_PER_DAY = 16
EXCLUDED_TIMES = {'09:30', '18:00'}
CAPITAL = 100000.0  # 100k TL Sermaye

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

def extract_signals(ret_pivot, K=10, N=1, strategy_name="Strateji", is_b2=False):
    dates = sorted(ret_pivot.index.get_level_values('date').unique())
    tickers = np.array(ret_pivot.columns)
    
    results = []
    
    for d_idx, d in enumerate(dates):
        if d_idx < K:
            continue
            
        # Sinyal Sadece 16. periyotta (17:30 kapanışında) üretilir
        if (d, 16) not in ret_pivot.index:
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
            
            # Seçimi belirle
            r_d = ret_pivot.loc[(d, 16)].values[0]
            valid_mask = np.isfinite(sig_d) & np.isfinite(r_d)
            if valid_mask.sum() >= (N * 2):
                valid_indices = np.where(valid_mask)[0]
                sorted_valid = valid_indices[np.argsort(sig_d[valid_indices])]
                
                bottom_indices = sorted_valid[:N] # SHORT
                top_indices = sorted_valid[-N:]   # LONG
                
                long_stocks = tickers[top_indices]
                short_stocks = tickers[bottom_indices]
                
                # İleriye dönük getiriyi hesapla
                holding_days = 10 if is_b2 else 1
                
                trade_profit_tl = 0.0
                if d_idx + 1 < len(dates):
                    # Kalan gün sayısını aşmamak için güvenli bitiş indeksi
                    end_idx = min(d_idx + 1 + holding_days, len(dates))
                    period_dates = dates[d_idx + 1 : end_idx]
                    
                    cum_ret_long = 0.0
                    cum_ret_short = 0.0
                    
                    for p_date in period_dates:
                        try:
                            # O günkü kümülatif getiri
                            day_ret_long = ret_pivot.loc[p_date][long_stocks].sum(axis=0).mean()
                            day_ret_short = ret_pivot.loc[p_date][short_stocks].sum(axis=0).mean()
                            cum_ret_long += day_ret_long
                            cum_ret_short += day_ret_short
                        except:
                            pass
                            
                    # Toplam getiri (Long getiri eksi Short getiri)
                    trade_return = cum_ret_long - cum_ret_short
                    
                    # 3.5 BPS Entry + 3.5 BPS Exit on 100k gross exposure = 70 TL
                    commission_tl = 70.0
                    trade_profit_tl = ((CAPITAL / 2.0) * trade_return) - commission_tl
                    
                label_kz = f"{holding_days} Günlük K/Z" if is_b2 else "Ertesi Gün K/Z"
                    
                results.append({
                    'Tarih': d,
                    'Strateji': strategy_name,
                    'Alınanlar (Long)': ", ".join(long_stocks),
                    'Açığa Satılanlar (Short)': ", ".join(short_stocks),
                    'Sermaye': f"{CAPITAL:,.0f} TL",
                    'K/Z Dönemi': label_kz,
                    'K/Z (TL)': f"{trade_profit_tl:,.0f} TL"
                })

    return results[-15:]

if __name__ == "__main__":
    print("Veriler yükleniyor...")
    ret_pivot = load_data()
    
    print("Sinyaller hesaplanıyor...")
    res_a1_1x1 = extract_signals(ret_pivot, K=10, N=1, strategy_name="Strateji 1 (A1 1x1)", is_b2=False)
    res_b2_5x5 = extract_signals(ret_pivot, K=10, N=5, strategy_name="Strateji 2 (B2 5x5)", is_b2=True)
    res_b2_1x1 = extract_signals(ret_pivot, K=10, N=1, strategy_name="Strateji 3 (B2 1x1)", is_b2=True)
    
    import json
    
    output = {
        "A1_1x1": res_a1_1x1,
        "B2_5x5": res_b2_5x5,
        "B2_1x1": res_b2_1x1
    }
    
    with open('last_15_trades.json', 'w') as f:
        json.dump(output, f, indent=4, default=str)
    
    print("Bitti. JSON kaydedildi.")
