# Northflank Üzerinde Bot Kurulum Rehberi

Northflank, GitHub deponuzdaki Dockerfile'ı kullanarak kodunuzu bir "Cron Job" (Zamanlanmış Görev) olarak sorunsuzca çalıştırmanıza olanak sağlayan modern bir bulut platformudur.

Sizin için sisteme bir `Dockerfile` ve `run_and_push.sh` dosyası ekledim. Bu sayede Northflank botu çalıştırdıktan sonra yeni portföy verisini otomatik olarak GitHub deponuza geri gönderecek!

---

## Adım 1: GitHub Personal Access Token (PAT) Alma
Botun GitHub'a veri yazabilmesi için şifre yerine geçen bir "Token" almanız gerekiyor.
1. GitHub'a girin: **Settings -> Developer Settings -> Personal access tokens -> Tokens (classic)**
2. **"Generate new token (classic)"** butonuna basın.
3. İsim olarak `NorthflankBot` verin. Expiration (Süre) kısmını `No expiration` yapabilirsiniz.
4. Alt kısımdaki yetkilerden **`repo`** yazan ana kutucuğu işaretleyin.
5. En alttan oluşturun ve çıkan uzun şifreyi (ghp_ ile başlar) **mutlaka bir yere kopyalayın** (bir daha göremeyeceksiniz).

---

## Adım 2: Northflank'e Projeyi Ekleme
1. [Northflank.com](https://northflank.com/)'a giriş yapın ve bir Proje oluşturun (Create Project).
2. Projenin içine girip sağ üstten **Create -> Job** seçeneğine tıklayın.
3. Job Type olarak **"Cron Job"** seçin.
4. Cron Schedule (Zamanlama) kısmına şu ifadeyi yapıştırın:
   `*/30 7-15 * * 1-5`
   *(Bu, hafta içi UTC ile 07:00-15:00 yani TSİ 10:00-18:00 arası her 30 dakikada bir çalıştırır)*
5. **Source** kısmında "Version Control" seçin ve GitHub hesabınızı bağlayarak `INTRADAY_BIST_TRADE_BOT` reponuzu seçin.
6. **Build Type** olarak "Dockerfile" seçili kalsın (Northflank otomatik bulacaktır).
7. Job'ı oluşturun (Create Job).

---

## Adım 3: Secret (Çevre Değişkeni) Ekleme
Oluşturduğunuz Job'ın detay sayfasına girin. Soldaki menüden **"Environment"** veya **"Secrets"** sekmesine tıklayın.
Aşağıdaki 3 değişkeni (Environment Variables) tek tek ekleyin:

1. Anahtar: `GITHUB_PAT`
   Değer: *(Az önce 1. Adımda kopyaladığınız ghp_ ile başlayan uzun şifre)*
2. Anahtar: `GITHUB_USER`
   Değer: `ysfgkcbnr` *(GitHub kullanıcı adınız)*
3. Anahtar: `GITHUB_REPO`
   Değer: `ysfgkcbnr/INTRADAY_BIST_TRADE_BOT`

Bunları kaydedin. 

---

## Adım 4: İlk Deneme
Northflank panelinde Job sayfanıza dönün ve sağ üstten **"Run Job"** (veya Execute) butonuna manuel olarak basıp test edin.

Aşağıdaki terminal/log ekranında botun çalıştığını ve son satırlarda `Auto Update: Paper Portfolio State from Northflank` mesajıyla dosyayı GitHub'a başarıyla Push ettiğini göreceksiniz. Artık her şey otomatik!
