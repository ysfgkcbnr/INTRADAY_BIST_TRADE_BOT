# Oracle Cloud (Ubuntu VM) Üzerinde Bot Kurulum Rehberi

GitHub Actions'taki kısıtlamaları tamamen aşıp kendi sanal sunucunuzda (VPS) özgürce çalışmak harika bir karar! Oracle Cloud'un ücretsiz (Always Free) Ubuntu sunucuları bu iş için biçilmiş kaftandır. 

Aşağıdaki adımları takip ederek botunuzu 7/24 çalışacak şekilde Oracle Cloud'a kurabilirsiniz.

---

## Adım 1: Sunucuya (VM) Bağlanma

Oracle Cloud panelinden oluşturduğunuz Ubuntu (veya Oracle Linux) sunucusuna terminal (Mac/Linux) veya PuTTY (Windows) üzerinden SSH ile bağlanın:

> [!NOTE]
> Sunucuyu oluştururken indirdiğiniz özel anahtar (private key) dosyasının adını `anahtar_dosyasi.key` ve sunucu IP adresinizi de `SUNUCU_IP_ADRESI` olarak varsayıyoruz.

```bash
# Mac/Linux terminalinde:
chmod 400 anahtar_dosyasi.key
ssh -i anahtar_dosyasi.key ubuntu@SUNUCU_IP_ADRESI
```

*(Eğer Oracle Linux kurduysanız kullanıcı adı `ubuntu` yerine `opc` olacaktır).*

---

## Adım 2: Gerekli Paketlerin Kurulumu

Sunucuya bağlandıktan sonra Python, Git ve diğer sistem araçlarını kuralım:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-pip python3-venv cron
```

---

## Adım 3: Projeyi Klonlama ve Kurulum

GitHub'daki deponuzu (reposunu) sunucuya indirin:

```bash
# Sizin reponuzu klonluyoruz:
git clone https://github.com/ysfgkcbnr/INTRADAY_BIST_TRADE_BOT.git

# Proje klasörüne giriyoruz:
cd INTRADAY_BIST_TRADE_BOT
```

Daha temiz bir yapı için Python sanal ortamı (venv) oluşturup kütüphaneleri yükleyelim:

```bash
python3 -m venv venv
source venv/bin/activate

# Gereksinimleri (numpy, pandas, yfinance vb.) yüklüyoruz:
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Adım 4: Botu Manuel Test Etme

Sistemin düzgün çalışıp çalışmadığını (verileri çekip çekmediğini) test etmek için strateji kodunu bir kez el ile çalıştırın:

```bash
# Hala proje klasöründe (INTRADAY_BIST_TRADE_BOT) ve venv aktifken:
python strateji_3_test_b2_1x1.py --run

# Raporu görmek için:
python strateji_3_test_b2_1x1.py --report
```
Eğer hatalı bir durum yoksa terminalde cüzdan bakiyesini ve alınan/satılan hisseleri göreceksiniz.

---

## Adım 5: Otomasyon (Cron Job) Kurulumu

Sunucunun hafta içi her gün 10:00 ile 18:00 arasında, her 30 dakikada bir otomatik olarak bu kodu çalıştırmasını sağlamak için **Cron** kullanacağız.

1. Cron editörünü açın:
```bash
crontab -e
```
*(Eğer editör seçmenizi isterse klavyeden `1` tuşuna basıp `nano` editörünü seçin).*

2. Açılan dosyanın en alt satırına inip şu kodu yapıştırın:
> [!IMPORTANT]  
> Sunucunun saati genellikle UTC'dir. Türkiye saati (TSİ) UTC'den 3 saat ileridedir. TSİ 10:00 - 18:00 aralığı, UTC'de 07:00 - 15:00 aralığına denk gelir. Aşağıdaki kod UTC'ye göre yazılmıştır.

```bash
# Hafta içi (1-5) UTC saat 07:00-15:00 arasında her 30 dakikada bir çalıştır
*/30 7-15 * * 1-5 cd /home/ubuntu/INTRADAY_BIST_TRADE_BOT && /home/ubuntu/INTRADAY_BIST_TRADE_BOT/venv/bin/python strateji_3_test_b2_1x1.py --run >> /home/ubuntu/INTRADAY_BIST_TRADE_BOT/bot.log 2>&1
```

3. Dosyayı kaydetmek için klavyede sırasıyla şu tuşlara basın:
   - `CTRL + O` (Kaydetmek için)
   - `Enter` (Onaylamak için)
   - `CTRL + X` (Çıkmak için)

---

## Adım 6: Logları Takip Etme

Tebrikler! Botunuz artık Oracle Cloud sunucusunda, tamamen arka planda, sessizce ve sizin bilgisayarınız kapalı olsa bile çalışıyor.

Botun arka planda her 30 dakikada bir neler yaptığını canlı izlemek için terminale şu komutu yazabilirsiniz:

```bash
tail -f bot.log
```

*(Takip modundan çıkmak için `CTRL + C` yapabilirsiniz).*
