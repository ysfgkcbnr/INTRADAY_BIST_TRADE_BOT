# BIST 100 Intraday Periodicity Live Paper Trading Bots

This repository contains algorithmic trading bots that exploit intraday return periodicity in the Borsa Istanbul (BIST 100) market. Built upon the empirical framework of Heston, Korajczyk, and Sadka (2010), the models identify predictable 30-minute return patterns, specifically focusing on the powerful 17:30 market closing bar.

After conducting exhaustive backtesting and rigorous statistical validation (Nested Walk-Forward Optimization, CSCV Probability of Backtest Overfitting, and Newey-West HAC tests), we identified 3 winning quantitative strategies. These strategies have now been packaged as standalone live paper trading bots capable of trading virtual capital in real-time.

---

## 🏆 The Champion Strategy Bots (Limit-Up/Down Protected)

The repository includes independent Python bots. Each bot runs on a starting virtual equity of **100,000 TL**, simulating a **2.0x Notional Exposure** (100k Spot Buy + 100k VIOP Short) with a realistic transaction cost of **2.0 BPS (0.020%)**. Importantly, their performance has been validated strictly under **Borsa Istanbul's 9% Limit Up/Down rules**, blocking any fictitious profits.

### 1. `strateji_1_test_a1_1x1.py` (The 17:30 Sniper)
- **Logic:** Only trades the highly predictable 17:30 closing bar. It goes Long on the Top 1 stock with the highest periodicity signal and Short on the Bottom 1 stock.
- **Performance:** Achieved **+2.75 Net Sharpe Ratio**, **+49.84% Net Total Return**, and zero overfitting risk (PBO 9.09%).

### 2. `strateji_3_test_b2_1x1.py` (The Ultimate Record Champion)
- **Logic:** Combines the 10-day 1/10 overlapping holding structure with the concentrated extreme signal (Top 1 Long / Bottom 1 Short).
- **Performance:** The absolute winner. Achieved **+184.22% Net Total Return (4.5 years)**, **+26.87% Net CAGR**, **1.07 Net Sharpe**, and is statistically significant at a 95%+ confidence level (Newey-West p < 0.05).

---

## 🚀 Live Paper Trading Mechanics & Architecture (Fly.io + TradingView)

These scripts operate as a robust webhook-based live trading bot that no longer relies on slow public APIs. Instead, it uses **TradingView** and a local **SQLite Database**.

1. **TradingView Webhooks (`tradingview_datafeed.pine`):** A custom Pine Script indicator runs on TradingView and sends live price data via webhooks exactly at **17:30** and **18:00**.
2. **Fly.io Webhook Server (`app.py`):** A lightweight Flask server listens for these incoming webhooks on a cloud container (Fly.io).
3. **Database (`market_data.db`):** Intraday historical prices for 28 liquid BIST 100 stocks are continuously stored in a local SQLite database, updated periodically by background tasks (`db_updater.py`) using `tvDatafeed`.
4. **Execution Workflow:** 
   - **At 17:30:** The server receives the webhook, saves open prices, triggers **Test A1 Entry**, and triggers **Test B2 Rebalance**.
   - **At 18:00:** The server receives the webhook, saves closing prices, and triggers **Test A1 Exit**.
5. **Portfolio Persistence:** The bots calculate historical $K=10$ periodicity signals, dynamically rebalance virtual equity, deduct commissions, and save their states into local JSON files (e.g., `portfolio_b2_1x1.json`). The server then automatically commits these files back to this GitHub repository to ensure no state is lost.

---

## ⚙️ Installation & Usage

### 1. Install Dependencies
Make sure you have Python 3.10+ installed. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Initialize Database & Run the Server
```bash
# Optional: Seed the database from your local CSV archives
python3 init_db.py

# Run the webhook server locally
python3 app.py
```

---

## 🔬 Statistical Validation & Research (Tavan/Taban Limit Filtered)

The quantitative edge of these bots has been rigorously validated by an institutional-grade backtesting engine (`comprehensive_limit_backtest.py`), heavily modified to account for Borsa Istanbul's strict **9% limit up/down rules**:
- **BIST 9% Limit Filter:** The engine strictly prevents trading on stocks locked at limit-up/down prices relative to the previous day's close. This eliminates fictitious profits.
- **CSCV (Probability of Backtest Overfitting):** Proves the strategies are not overfit to noise (e.g., PBO score of ~21% for Test B2).
- **Nested Walk-Forward Optimization:** Validates that the models sustain strong positive returns on completely unseen rolling out-of-sample data.
- **Newey-West HAC Tests:** Confirms the statistical significance of the returns, adjusting for heteroskedasticity and autocorrelation.

**Disclaimer:** This project is for educational and quantitative research purposes only. The bots trade entirely with virtual "paper" money. Do not connect this logic to live brokerage accounts without extensive real-time forward testing.
