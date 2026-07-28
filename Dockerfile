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

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
