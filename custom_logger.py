import os
import logging
import sys

def get_app_logger(name="BIST_BOT"):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Konsol (Console) Çıktısı için ayar
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Logtail (Better Stack) Çıktısı için ayar
        logtail_token = os.environ.get("LOGTAIL_TOKEN")
        if logtail_token:
            try:
                from logtail import LogtailHandler
                logtail_handler = LogtailHandler(source_token=logtail_token, host="in.eu.logs.betterstack.com")
                logtail_handler.setLevel(logging.INFO)
                logger.addHandler(logtail_handler)
                logger.info("Logtail başarıyla bağlandı!")
            except ImportError:
                logger.error("logtail-python kütüphanesi bulunamadı! Lütfen gereksinimleri yükleyin.")
            except Exception as e:
                logger.error(f"Logtail bağlanırken hata: {e}")
                
    return logger

# Ortak kullanılacak logger objesi
logger = get_app_logger()
