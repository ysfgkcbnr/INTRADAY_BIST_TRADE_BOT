from datetime import timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
TZ_ISTANBUL = timezone(timedelta(hours=3))
try:
    scheduler = BackgroundScheduler(timezone=TZ_ISTANBUL)
    print("No error")
except Exception as e:
    print("Error:", e)
