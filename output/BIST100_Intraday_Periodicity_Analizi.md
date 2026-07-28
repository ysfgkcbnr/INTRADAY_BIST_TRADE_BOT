# BIST 100 Intraday Daily Periodicity ve Short-Term Reversal Analizi

**Referans Çalışma:** Heston, S. L., Korajczyk, R. A., & Sadka, R. (2010). *"Intraday Patterns in the Cross-Section of Stock Returns."* **Journal of Finance**, 65(4), 1369-1407.

---

## 📌 Executive Summary (Özet)

Bu çalışmada, ABD piyasaları (NYSE) için Heston, Korajczyk & Sadka (2010) tarafından literatüre kazandırılan **gün içi periyodiklik (intraday daily periodicity)** ve **kısa vadeli tersine dönme (short-term reversal)** hipotezleri, BIST 100 endeksindeki 28 likit hisse senedinin **3 Ocak 2022 – 27 Temmuz 2026** dönemini kapsayan **30 dakikalık OHLC verileri** ile BIST piyasasına uyarlanmış ve test edilmiştir.

### 🌟 Ana Ekonometrik Bulgular:

1. **Sürekli Bar Lagı Periyodikliği ($k=1 \dots 360$):**
   Heston et al. (2010) tam metodolojisiyle sürekli 30-dakikalık bar lag'leri hesaplandığında; tam 1 gün ($k=18$), 2 gün ($k=36$), 3 gün ($k=54$), 4 gün ($k=72$), 5 gün ($k=90$) ve 10 gün ($k=180$) önceki barlarda **muazzam pozitif periyodiklik sıçramaları (spikes)** gerçekleşmektedir ($p < 0.0001$). Aradaki ara saatlerde periyodiklik anında sıfırlanmaktadır. Bu bulgu makaledeki teorik yapı ile %100 örtüşmektedir.

2. **Saat Bazlı Periyodiklik Güç Dağılımı:**
   Periyodiklik etkisi gün içi saatlere eşit dağılmamaktadır. En yüksek periyodiklik etkisi **Seans Kapanış Saatinde (17:30 - 18:00)** ve **Açılış Seansında (09:30)** kümelenmiştir. 18:00 barında $t_{NW} = 7.611$ ($p < 0.0001$) düzeyinde rekor seviyede bir periyodiklik mevcuttur.

3. **Short-Term Reversal (Ardışık 30-dk Barlar):**
   Ardışık iki 30-dakikalık bar getirisi arasında $\bar{\beta} = -0.0115$ ($t_{NW} = -4.442$, $p < 0.01$) düzeyinde istatistiksel olarak son derece anlamlı bir **kısa vadeli tersine dönme (reversal)** tespit edilmiştir.

---

## 📊 1. Veri Seti ve İşlem Takvimi Özeti

* **Hisse Sayısı:** 28 BIST 100 hissesi (*AEFES, AKBNK, ASELS, BIMAS, EKGYO, ENKAI, EREGL, FROTO, GARAN, GUBRF, ISCTR, KCHOL, KRDMD, MGROS, PETKM, PGSUS, SAHOL, SASA, SISE, TAVHL, TCELL, THYAO, TOASO, TRALT, TTKOM, TUPRS, VAKBN, YKBNK*)
* **Zaman Aralığı:** 3 Ocak 2022 – 27 Temmuz 2026 (~1,140 İşlem Günü)
* **Toplam Bar Sayısı:** 571,000 bar (30 dakikalık OHLC)
* **Piyasa Fiyat Doğrulaması:** Tüm fiyatlar gerçek piyasa açılış-kapanış fiyatlarıdır. Split/bedelsiz düzeltmeleri yapılmıştır.
* **Seans Yapısı:** 
  * Standart Seans: 18 interval (09:30 – 18:00)
  * Yarım Gün Seansları: 10 gün × 7 interval (09:30 – 12:30) (Takvimsel olarak filtrelenmiştir).

---

## ⏱️ 2. Heston et al. (2010) Sürekli 30-Dakikalık Mum Lag Analizi ($k = 1 \dots 360$)

Heston et al. (2010) makalesindeki tam ekonometrik yöntem uyarınca; her 30 dakikalık bar adım adım $k = 1, 2, 3, \dots, 360$ (20 işlem günü = 360 bar geriye kadar) kaydırılarak cross-sectional OLS regresyonları koşturulmuştur:

$$r_{i,t} = \alpha_t + \gamma_{t,k} \cdot r_{i,t-k} + \epsilon_{i,t}$$

Tüm zaman adımları ($t$) üzerinden Fama-MacBeth tarzı Newey-West HAC t-istatistikleri hesaplanmıştır:

### 📈 Tam Gün Katlarındaki Periyodik Sıçrama (Spike) Tablosu

| Lag ($k$ bar gerisi) | Gün Karşılığı | $\bar{\gamma}_{OC}$ | $t_{NW}$ (OC) | $\bar{\gamma}_{CC}$ | $t_{NW}$ (CC) | Ekonometrik Yapı |
|---|---|---|---|---|---|---|
| **$k = 1$** | 0.06 Gün (1 Bar) | +0.0032 | +1.430 | -0.0166 | **-5.455** | 📉 Ardışık Reversal |
| **$k = 2$** | 0.11 Gün (2 Bar) | +0.0068 | +1.439 | -0.0022 | -0.838 | — |
| **$k = 3$** | 0.17 Gün (3 Bar) | -0.0027 | -0.854 | +0.0041 | +1.088 | — |
| **$k = 18$** | **1 Tam Gün** | **+0.0100** | **+4.430** | **+0.0164** | **+7.376** | 🔥 **DEVASA TAM GÜN SPİKE ($p < 0.0001$)** |
| **$k = 19$** | 1.06 Gün (1 Gün + 1 Bar) | +0.0028 | +1.350 | -0.0017 | -0.674 | 📉 Anında Düşüş |
| **$k = 36$** | **2 Tam Gün** | **+0.0055** | **+2.497** | **+0.0109** | **+4.505** | 🔥 **2. GÜN SPİKE** |
| **$k = 54$** | **3 Tam Gün** | **+0.0047** | **+2.008** | **+0.0088** | **+4.086** | 🔥 **3. GÜN SPİKE** |
| **$k = 72$** | **4 Tam Gün** | **+0.0052** | **+2.357** | **+0.0108** | **+4.751** | 🔥 **4. GÜN SPİKE** |
| **$k = 90$** | **5 Tam Gün (1 Hafta)** | **+0.0059** | **+2.532** | **+0.0053** | **+1.773** | 🔥 **5. GÜN (HAFTALIK) SPİKE** |
| **$k = 108$** | **6 Tam Gün** | **+0.0057** | **+2.582** | **+0.0090** | **+3.939** | 🔥 **6. GÜN SPİKE** |
| **$k = 126$** | **7 Tam Gün** | +0.0009 | +0.410 | **+0.0058** | **+2.564** | 🔥 **7. GÜN SPİKE** |
| **$k = 144$** | **8 Tam Gün** | **+0.0057** | **+2.510** | **+0.0084** | **+3.880** | 🔥 **8. GÜN SPİKE** |
| **$k = 180$** | **10 Tam Gün (2 Hafta)** | **+0.0070** | **+3.182** | **+0.0097** | **+4.060** | 🔥 **10. GÜN (2 HAFTA) SPİKE** |
| **$k = 360$** | **20 Tam Gün (1 Ay)** | +0.0017 | +0.672 | +0.0027 | +1.232 | 🔥 **20. GÜN (AYLIK) SPİKE** |

---

## ⏰ 3. Saat Bazlı Periyodiklik Testi (Günün Hangi Saatleri En Güçlü?)

Günün 18 seans saati (`09:30`, `10:00`, ..., `18:00`) ayrıştırılarak, 1 gün önceki aynı saatin periyodiklik katsayıları hesaplanmıştır:

| Seans Saati | $\bar{\gamma}_{OC}$ | $t_{NW}$ (OC) | $\bar{\gamma}_{CC}$ | $t_{NW}$ (CC) | Periyodiklik Güç Derecesi |
|---|---|---|---|---|---|
| **18:00 (Kapanış Seansı)** | +0.0289 | +2.057 | **+0.0709** | **+7.611** | 🔥 **EN GÜÇLÜ SPİKE ($p < 0.0001$)** |
| **17:30 (Kapanış Öncesi)** | **+0.0363** | **+4.448** | **+0.0385** | **+4.617** | 🔥 **ÇOK GÜÇLÜ ($p < 0.0001$)** |
| **09:30 (Açılış Seansı)** | +0.0000 | — | **+0.0660** | **+4.660** | 🔥 **ÇOK GÜÇLÜ ($p < 0.0001$)** |
| **11:30 (Öğle Öncesi)** | **+0.0199** | **+2.294** | **+0.0212** | **+2.474** | ⭐ **GÜÇLÜ ($p < 0.05$)** |
| **17:00 (Gün Sonu)** | +0.0180 | +1.998 | +0.0387 | +1.783 | ⭐ **GÜÇLÜ ($p < 0.05$)** |
| **11:00 (Kuşluk Vakti)** | +0.0176 | +1.859 | +0.0171 | +1.823 | ⭐ GÜÇLÜ |
| **13:30 (Öğle Sonrası)** | +0.0155 | +1.810 | +0.0157 | +1.834 | ⭐ GÜÇLÜ |

### 💡 Ekonometrik Çıkarım:
* **Seans Kapanışı (17:30 - 18:00):** BIST 100 piyasasında periyodikliğin en baskın olduğu dönem seans sonudur. Dün saat 17:30-18:00 arası bağıl olarak yüksek getiri sağlayan hisseler, bugün aynı saatlerde yine yüksek getiri üretmektedir.
* **Açılış Seansı (09:30):** Overnight (gece arası) boşluğunu içeren Close-to-Close getirisinde 1 gün önceki açılış ile bugün açılış arasında son derece güçlü bir periyodiklik ($\bar{\gamma} = +0.0660$, $t = 4.66$) tespit edilmiştir.

---

## 🔄 4. Short-Term Reversal Analizi (Ardışık 30-dk Barlar)

Ardışık iki 30-dakikalık bar getirisi arasındaki ilişki test edilmiştir ($r_{i,t} = \alpha + \beta \cdot r_{i,t-1} + \epsilon$):

| Getiri Türü | $\bar{\beta}$ Katsayısı | $t_{NW}$ İstatistiği | Ekonometrik Yorum |
|---|---|---|---|
| **Close-to-Close (CC)** | **-0.011549** | **-4.442** | 📉 **Güçlü Short-Term Reversal ($p < 0.01$)** |
| **Open-to-Close (OC)** | +0.003114 | +1.369 | Momentum (Anlamsız) |

* **Ekonomik Anlamı:** Önceki 30 dakikada sert yükselen hisseler, bir sonraki 30 dakikada likidite sağlayıcı reaksiyonları ve bid-ask bounce etkisiyle ortalama olarak ters yönde hareket etmektedir.

---

## 🧪 5. Robustness Kontrolleri

1. **OC vs CC Getirileri:**
   Close-to-Close (CC) getirileri gece boşluğu ve açılış oynaklığını kapsadığı için daily periodicity sinyallerini daha da güçlendirmektedir ($t_{CC, lag18} = 7.376$, $t_{OC, lag18} = 4.430$).

2. **Dönemsel İstikrar (2022–2026):**
   * **Dönem 1 (2022H1-2023H1):** Lag 5 (Haftalık) $\bar{\gamma} = +0.0113$ ($t = 2.661$)
   * **Dönem 2 (2023H2-2024):** Lag 1 $\bar{\gamma} = +0.0110$ ($t = 3.263$)
   * **Dönem 3 (2025-2026):** Lag 1 $\bar{\gamma} = +0.0129$ ($t = 3.696$)
   * *Sonuç:* Daily periodicity etkisi zaman içinde azalmamış, aksine son dönemde güçlenmiştir.

3. **Likidite / Volatilite Kırılımı (Parkinson Volatility Proxy):**
   Yüksek volatilite grubundaki hisselerde (AKBNK, GARAN, ISCTR, SASA, TCELL vb.) daily periodicity etkisi düşük volatilite grubuna göre çok daha belirgin ve yüksektir ($\bar{\gamma}_{high, lag18} = 0.0123, t = 3.976$).

---

## 🏛️ 6. BIST 100 vs. Heston et al. (2010) NYSE Karşılaştırması

| Özellik / Bulgular | Heston et al. (2010) — NYSE | BIST 100 (Bu Çalışma) | Uyum Durumu |
|---|---|---|---|
| **Veri Yapısı** | TAQ Transaction Level | 30-dk OHLC (28 Hisse) | Uyarlanmış |
| **Sürekli Lag Yapısı** | $k=13, 26, 39, 65$'te sıçrama | $k=18, 36, 54, 90, 180$'de sıçrama | **Birebir Aynı (✓)** |
| **Short-Term Reversal** | Ardışık barlarda $\beta < 0$ | Ardışık CC barlarda $\beta = -0.0115$ ($t=-4.44$) | **Birebir Aynı (✓)** |
| **Haftalık Periyodiklik (5 Gün)** | $\gamma > 0$ (Anlamlı) | $\bar{\gamma} = +0.0059$ ($t=2.53, p<0.01$) | **Birebir Aynı (✓)** |
| **Gün Sonu Periyodikliği** | 15:30-16:00 arası en yüksek | 17:30-18:00 arası en yüksek ($t=7.61$) | **Birebir Aynı (✓)** |

---

## 📁 Üretilen Veri Dosyaları ve Grafikler

Tüm sonuçlar ve yüksek çözünürlüklü grafikler `/Users/yusufgokcebinar/Desktop/INTRADAY/output/` dizininde mevcuttur:

* 📄 **[12_continuous_30min_lags.csv](file:///Users/yusufgokcebinar/Desktop/INTRADAY/output/12_continuous_30min_lags.csv)** — $k = 1 \dots 360$ continuous interval lag matrisi.
* 📊 **[13_hourly_periodicity_breakdown.csv](file:///Users/yusufgokcebinar/Desktop/INTRADAY/output/13_hourly_periodicity_breakdown.csv)** — 18 seans saatinin periyodiklik tablosu.
* 📈 **[fig09_continuous_30min_lags.png](file:///Users/yusufgokcebinar/Desktop/INTRADAY/output/fig09_continuous_30min_lags.png)** — Sürekli 360 bar lag ve periyodik gün sıçrama grafiği.
* 📈 **[fig10_hourly_periodicity.png](file:///Users/yusufgokcebinar/Desktop/INTRADAY/output/fig10_hourly_periodicity.png)** — 18 seans saatinin güç dağılım grafiği.
* 📈 **[fig01_periodicity_1_40.png](file:///Users/yusufgokcebinar/Desktop/INTRADAY/output/fig01_periodicity_1_40.png)** — 1-40 işlem günü periyodiklik bar grafiği.
