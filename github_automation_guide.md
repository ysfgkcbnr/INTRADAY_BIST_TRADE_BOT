# GitHub Actions 24/7 Otomatik Canlı Sanal Ticaret (Paper Trading) Rehberi

Bu rehber, oluşturduğumuz **3 Şampiyon Strateji Botunu** GitHub'a yükleyip BIST borsa saatlerinde (10:00 - 18:00) **GitHub Actions** ile 24 saat **ÜCRETSİZ** ve otomatik çalışır hale getirmeniz için adım adım hazırlanmıştır.

---

## 📁 1. Oluşturulan 3 Bağımsız Bot Dosyası

Dizininizde 3 farklı strateji için 3 ayrı bağımsız Python botu hazırlanmıştır:

1. **`strateji_1_test_a1_1x1.py`**: Sadece 17:30 kapanış barında 1 Long / 1 Short işlemi yapan hızlı bot.
2. **`strateji_2_test_b2_5x5.py`**: 10 gün çakışmalı 5 Long / 5 Short taşıma botu.
3. **`strateji_3_test_b2_1x1.py`**: **+184.2% Rekor Kâr üreten** 10 gün çakışmalı 1 Long / 1 Short taşıma botu.

---

## 🚀 2. GitHub Actions Otomasyon Kurulumu (Adım Adım)

### Adım A: Projenizi GitHub'a Yükleyin
Terminalde projenizin bulunduğu dizinde aşağıdaki komutları çalıştırarak projenizi GitHub reponuza yükleyin:

```bash
git init
git add .
git commit -m "BIST Intraday Periodicity Live Paper Trading Bots"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADI/INTRADAY.git
git push -u origin main
```

---

### Adım B: GitHub Actions İş Akışı Dosyası Oluşturma
Proje dizininizde `.github/workflows/paper_trading.yml` yolunda aşağıdaki dosyayı oluşturun:

```yaml
name: BIST Intraday Live Paper Trading

on:
  schedule:
    # Hafta içi her gün saat 07:00 - 15:00 UTC (TSİ 10:00 - 18:00) arası her 30 dakikada bir çalışır
    - cron: '*/30 7-15 * * 1-5'
  workflow_dispatch: # GitHub arayüzünden manuel tetikleme butonu

jobs:
  trade:
    runs-on: ubuntu-latest
    steps:
      - name: Repoyu Klonla
        uses: actions/checkout@v3

      - name: Python Kurulumu
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Bağımlılıkları Yükle
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas yfinance scipy statsmodels

      - name: Strateji 3 (Rekor Şampiyon B2 1x1) Canlı Sinyal ve Rebalance
        run: |
          python strateji_3_test_b2_1x1.py --run

      - name: Sanal Portföy Durumunu Kaydet (Commit & Push)
        run: |
          git config --global user.name "GitHub Actions Bot"
          git config --global user.email "actions@github.com"
          git add portfolio_*.json
          git diff --quiet && git diff --staged --quiet || git commit -m "Auto Update: Paper Portfolio State [skip ci]"
          git push
```

---

## 📊 3. Yerel Test ve Komut Kullanımları

Herhangi bir botu yerel bilgisayarınızda test etmek veya rapor almak için:

```bash
# Strateji 3 (Rekor Şampiyon) Canlı Güncelleme
python3 strateji_3_test_b2_1x1.py --run

# Sanal Portföy Raporu Görüntüleme
python3 strateji_3_test_b2_1x1.py --report

# Sanal Bakiyeyi 100.000 TL'ye Sıfırlama
python3 strateji_3_test_b2_1x1.py --reset
```
