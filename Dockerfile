FROM python:3.10-slim

# Sistem bağımlılıklarını (özellikle git ve cron) kur
RUN apt-get update && apt-get install -y git cron && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını kopyala
COPY . .

# Shell script'e çalışma izni ver
RUN chmod +x run_and_push.sh

# Cron ayarını oluştur (Hafta içi UTC 07:00-15:00 arası her saatin 15. ve 45. dakikasında çalışır)
# NOT: Çevre değişkenlerinin (Secrets) Cron içinde okunabilmesi için /etc/environment'a atılması gerekir.
# Bunu script çalışırken otomatik yapması için cron'u özel ayarlıyoruz.
RUN echo "15,45 7-15 * * 1-5 root env > /etc/environment && cd /app && ./run_and_push.sh >> /proc/1/fd/1 2>&1" > /etc/cron.d/bot-cron
RUN chmod 0644 /etc/cron.d/bot-cron
RUN crontab /etc/cron.d/bot-cron

# Container ayakta kalsın diye cron'u ön planda (foreground) çalıştır
CMD ["cron", "-f"]
