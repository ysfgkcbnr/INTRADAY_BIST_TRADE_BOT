#!/bin/bash
# Northflank veya herhangi bir Docker container'da botu çalıştırıp 
# JSON portföy durumunu GitHub'a geri kaydetmek için script

set -e # Hata olursa dur

# Cron environment doesn't have /usr/local/bin by default where python is installed in docker
export PATH="/usr/local/bin:$PATH"

# 1. Git kimlik bilgilerini yapılandır (Northflank secret'ı olarak GITHUB_EMAIL, GITHUB_USER ve GITHUB_PAT eklenmelidir)
git config --global user.email "${GITHUB_EMAIL:-bot@northflank.com}"
git config --global user.name "${GITHUB_USER:-NorthflankBot}"

# Güvenli klasör ayarı (Docker içinde bazen git klasör güvenliği hatası verebilir)
git config --global --add safe.directory /app

# Remote URL'yi GITHUB_PAT (Personal Access Token) ile güncelle ki şifresiz push yapabilsin
# GITHUB_REPO formatı: ysfgkcbnr/INTRADAY_BIST_TRADE_BOT
if [ -n "$GITHUB_PAT" ] && [ -n "$GITHUB_REPO" ]; then
    git remote set-url origin "https://${GITHUB_USER}:${GITHUB_PAT}@github.com/${GITHUB_REPO}.git"
fi

# 2. Çalıştırmadan önce reponun en güncel halini al
git pull --rebase origin main

# 3. Python strateji botunu çalıştır
python strateji_3_test_b2_1x1.py --run

# 4. Değişen JSON portföy dosyasını ve logları GitHub'a kaydet (Push)
git add portfolio_*.json signals_*.log

if ! git diff --staged --quiet; then
    git commit -m "Auto Update: Paper Portfolio State from Northflank [skip ci]"
    git push origin main
else
    echo "Değişiklik yok, push atlandı."
fi
