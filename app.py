from flask import Flask, request
import os
import json
import subprocess
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from db_updater import update_database_with_latest_bars
from custom_logger import logger

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TZ_ISTANBUL = timezone(timedelta(hours=3))

import threading

def process_1730_webhook():
    logger.info("Triggering A1 Strategy Entry...")
    res = subprocess.run(["python3", "strateji_1_test_a1_1x1.py", "--run_entry"], cwd=BASE_DIR, capture_output=True, text=True)
    if res.stdout: logger.info(f"A1 Entry Output:\n{res.stdout}")
    if res.stderr: logger.error(f"A1 Entry Error:\n{res.stderr}")

    logger.info("Triggering B2 Strategy...")
    res_b2 = subprocess.run(["python3", "strateji_3_test_b2_1x1.py", "--run"], cwd=BASE_DIR, capture_output=True, text=True)
    if res_b2.stdout: logger.info(f"B2 Output:\n{res_b2.stdout}")
    if res_b2.stderr: logger.error(f"B2 Error:\n{res_b2.stderr}")

    logger.info("Pushing changes to GitHub...")
    res_git = subprocess.run(["./run_and_push.sh"], cwd=BASE_DIR, capture_output=True, text=True)
    if res_git.stdout: logger.info(f"Git Push Output:\n{res_git.stdout}")

def process_1800_webhook():
    logger.info("Triggering A1 Strategy Exit...")
    res = subprocess.run(["python3", "strateji_1_test_a1_1x1.py", "--run_exit"], cwd=BASE_DIR, capture_output=True, text=True)
    if res.stdout: logger.info(f"A1 Exit Output:\n{res.stdout}")
    if res.stderr: logger.error(f"A1 Exit Error:\n{res.stderr}")

    logger.info("Pushing changes to GitHub...")
    res_git = subprocess.run(["./run_and_push.sh"], cwd=BASE_DIR, capture_output=True, text=True)
    if res_git.stdout: logger.info(f"Git Push Output:\n{res_git.stdout}")

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        if not data or 'time' not in data or 'prices' not in data:
            return "Invalid payload", 400

        time_str = data['time']
        prices = data['prices']
        now_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Webhook received for time: {time_str}")

        if time_str == "17:30":
            with open(os.path.join(BASE_DIR, 'open_prices.json'), 'w') as f:
                json.dump(prices, f)
            logger.info("Saved open_prices.json (used for A1 entry and B2 rebalance)")
            
            threading.Thread(target=process_1730_webhook).start()

        elif time_str == "18:00":
            with open(os.path.join(BASE_DIR, 'close_prices.json'), 'w') as f:
                json.dump(prices, f)
            logger.info("Saved close_prices.json (used for A1 exit)")
            
            threading.Thread(target=process_1800_webhook).start()

        else:
            logger.warning(f"Unknown time_str received: {time_str}")

        return "OK", 200

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return str(e), 500

@app.route('/', methods=['GET'])
def health_check():
    return "TradingView Webhook Server is running!", 200

# Initialize Scheduler
scheduler = BackgroundScheduler(timezone=TZ_ISTANBUL)
# Run every 30 minutes between 10:00 and 18:00
scheduler.add_job(
    update_database_with_latest_bars,
    'cron',
    day_of_week='mon-fri',
    hour='10-18',
    minute='0,30'
)
scheduler.start()

if __name__ == '__main__':
    # Run Flask directly for development, Gunicorn will be used for production
    app.run(host='0.0.0.0', port=8080)
