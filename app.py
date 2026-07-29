from flask import Flask, request
import os
import json
import subprocess
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from db_updater import update_database_with_latest_bars

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TZ_ISTANBUL = timezone(timedelta(hours=3))

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        if not data or 'time' not in data or 'prices' not in data:
            return "Invalid payload", 400

        time_str = data['time']
        prices = data['prices']
        now_str = datetime.now(TZ_ISTANBUL).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now_str}] Webhook received for time: {time_str}")

        if time_str == "17:00":
            # Save 17:00 prices (Open prices for the strategy)
            with open(os.path.join(BASE_DIR, 'open_prices.json'), 'w') as f:
                json.dump(prices, f)
            print("Saved open_prices.json")

            # Run A1 Strategy ENTRY
            print("Triggering A1 Strategy Entry...")
            subprocess.run(["python3", "strateji_1_test_a1_1x1.py", "--run_entry"], cwd=BASE_DIR)

            # Push changes to GitHub
            subprocess.run(["./run_and_push.sh"], cwd=BASE_DIR)

        elif time_str == "17:30":
            # Save 17:30 prices (Close prices for the strategy)
            with open(os.path.join(BASE_DIR, 'close_prices.json'), 'w') as f:
                json.dump(prices, f)
            print("Saved close_prices.json")

            # Run A1 Strategy EXIT
            print("Triggering A1 Strategy Exit...")
            subprocess.run(["python3", "strateji_1_test_a1_1x1.py", "--run_exit"], cwd=BASE_DIR)

            # Run B2 Strategy (Exit old overlapping portfolios and enter new ones)
            print("Triggering B2 Strategy...")
            subprocess.run(["python3", "strateji_3_test_b2_1x1.py", "--run"], cwd=BASE_DIR)

            # Push changes to GitHub
            subprocess.run(["./run_and_push.sh"], cwd=BASE_DIR)

        else:
            print(f"Unknown time_str received: {time_str}")

        return "OK", 200

    except Exception as e:
        print(f"Error processing webhook: {e}")
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
