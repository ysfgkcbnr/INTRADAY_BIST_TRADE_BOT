import json
import os

with open("last_15_trades.json") as f: 
    data = json.load(f)

md_path = "/Users/yusufgokcebinar/.gemini/antigravity-ide/brain/27470c13-7692-4633-8ce1-f40655bde211/son_15_islem.md"

with open(md_path, "w") as out:
    out.write("# 3 Strateji - Son 15 Gün İşlem (Backtest) Özeti\n\n")
    out.write("> [!NOTE]\n> Aşağıdaki tablolar, veri setinin son 15 gününde (en son sinyaller) her stratejinin **17:30 kapanışında hangi hisseleri seçtiğini**, 100.000 TL toplam sermayenin nasıl dağıtıldığını (50k Uzun / 50k Kısa) ve bu hisselerin ertesi günkü (T+1) hareketlerinden elde edilen tahmini brüt kar/zarar durumunu göstermektedir.\n\n")
    
    for strat, trades in data.items():
        out.write(f"## {strat}\n\n")
        out.write("| Tarih | Alınanlar (Long) | Satılanlar (Short) | Sermaye | K/Z Dönemi | K/Z (TL) |\n")
        out.write("|---|---|---|---|---|---|\n")
        total_pl = 0.0
        for t in trades:
            out.write(f"| {t['Tarih']} | {t['Alınanlar (Long)']} | {t['Açığa Satılanlar (Short)']} | {t['Sermaye']} | {t['K/Z Dönemi']} | {t['K/Z (TL)']} |\n")
            val_str = str(t['K/Z (TL)']).replace(' TL', '').replace(',', '')
            try:
                total_pl += float(val_str)
            except ValueError:
                pass
        
        out.write(f"| **TOPLAM (Son 15)** | | | | | **{total_pl:,.0f} TL** |\n\n")
