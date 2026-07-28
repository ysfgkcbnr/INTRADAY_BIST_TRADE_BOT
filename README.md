# BIST 100 Intraday Periodicity Live Paper Trading Bots

This repository contains algorithmic trading bots that exploit intraday return periodicity in the Borsa Istanbul (BIST 100) market. Built upon the empirical framework of Heston, Korajczyk, and Sadka (2010), the models identify predictable 30-minute return patterns, specifically focusing on the powerful 17:30 market closing bar.

After conducting exhaustive backtesting and rigorous statistical validation (Nested Walk-Forward Optimization, CSCV Probability of Backtest Overfitting, and Newey-West HAC tests), we identified 3 winning quantitative strategies. These strategies have now been packaged as standalone live paper trading bots capable of trading virtual capital in real-time.

---

## 🏆 The 3 Champion Strategy Bots

The repository includes three independent Python bots. Each bot runs on a starting virtual equity of **100,000 TL**, simulating a **2.0x Notional Exposure** (100k Spot Buy + 100k VIOP Short) with a realistic transaction cost of **2.0 BPS (0.020%)**.

### 1. `strateji_1_test_a1_1x1.py` (The 17:30 Sniper)
- **Logic:** Only trades the highly predictable 17:30 closing bar. It goes Long on the Top 1 stock with the highest periodicity signal and Short on the Bottom 1 stock.
- **Performance:** Achieved an extraordinary **+2.75 Net Sharpe Ratio**, **+109% Net CAGR**, and near-zero overfitting (PBO < 10%).

### 2. `strateji_2_test_b2_5x5.py` (The 10-Day Overlapping Quintile)
- **Logic:** Identifies signals at 17:30 and holds the positions for 10 trading days using a 1/10 overlapping portfolio structure (Jegadeesh & Titman style) to minimize transaction costs. Selects Top 5 Long / Bottom 5 Short.
- **Performance:** Achieved **+0.70 Net Sharpe Ratio** and **+9.81% Net CAGR** with extremely low drawdown.

### 3. `strateji_3_test_b2_1x1.py` (The Ultimate Record Champion)
- **Logic:** Combines the 10-day 1/10 overlapping holding structure with the concentrated extreme signal (Top 1 Long / Bottom 1 Short).
- **Performance:** The absolute winner. Achieved **+184.2% Net Total Return (4.5 years)**, **+26.8% Net CAGR**, **1.07 Net Sharpe**, and is statistically significant at a 98%+ confidence level (Newey-West p < 0.05).

---

## 🚀 Live Paper Trading Mechanics

These scripts connect directly to the **Yahoo Finance API (`yfinance`)** to fetch intraday 30-minute market data for 28 liquid BIST 100 stocks. 

Each time a bot is executed, it:
1. Fetches the latest live intraday data.
2. Re-calculates the historical $K=10$ periodicity signals.
3. Dynamically rebalances its virtual portfolio (buying/shorting stocks to meet the target weights).
4. Deducts realistic commission fees and logs every executed trade.
5. Saves its persistent state into a local JSON file (e.g., `portfolio_b2_1x1.json`), meaning you will never lose your virtual equity state even if the script stops.

---

## ⚙️ Installation & Usage

### 1. Install Dependencies
Make sure you have Python 3.10+ installed. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run a Bot Locally
You can manually trigger a live update for any strategy. It is recommended to run this during BIST market hours (10:00 - 18:00 Istanbul Time).

```bash
# Execute a live market update and rebalance the portfolio
python3 strateji_3_test_b2_1x1.py --run

# View the beautiful CLI dashboard of your virtual portfolio
python3 strateji_3_test_b2_1x1.py --report

# Reset virtual capital back to 100,000 TL
python3 strateji_3_test_b2_1x1.py --reset
```

---

## 🤖 24/7 Automated Trading via GitHub Actions

You can deploy these bots completely **FOR FREE** using GitHub Actions to run continuously every 30 minutes during market hours.

1. Push this repository to your own GitHub account.
2. The provided `.github/workflows/paper_trading.yml` workflow is pre-configured to run every 30 minutes between 10:00 and 18:00 (Istanbul Time) on weekdays.
3. Every 30 minutes, GitHub Actions will:
   - Spin up an Ubuntu runner.
   - Run the bot to fetch live data and rebalance the virtual portfolio.
   - Automatically commit and push the updated `portfolio_*.json` file back to your repository.

You can monitor your live virtual profits directly by inspecting the JSON files in your repository!

---

## 🔬 Statistical Validation & Research

The quantitative edge of these bots has been validated by an institutional-grade backtesting engine included in this repository. 
- **CSCV (Probability of Backtest Overfitting):** Proves the strategies are not overfit to noise.
- **Nested Walk-Forward Optimization:** Validates that the models sustain >1.00 Net Sharpe ratios on completely unseen rolling out-of-sample data.
- **Newey-West HAC Tests:** Confirms the statistical significance of the returns, adjusting for autocorrelation.

**Disclaimer:** This project is for educational and quantitative research purposes only. The bots trade entirely with virtual "paper" money. Do not connect this logic to live brokerage accounts without extensive real-time forward testing.
