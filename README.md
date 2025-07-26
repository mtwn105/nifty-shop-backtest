# Nifty Shop Backtesting

This project provides a comprehensive backtest for the "Nifty Shop" trading strategy on the Nifty 50 index, allowing for detailed performance analysis and strategy refinement.

## Strategy: Nifty Shop

The "Nifty Shop" strategy is a mean-reversion approach designed for beginners, focusing on systematically buying dips in Nifty 50 stocks.

### Entry Rules (checked daily at 3:20 PM)

- **Scan for Dips:** Identify the top five Nifty 50 stocks that are trading the furthest below their 20-day moving average (20 SMA).
- **Buy New Positions:** From this list, the script can buy up to a configurable number of new stocks (default is 1) that are not already in the portfolio.
- **Averaging Down:** If all top fallers are already held, the script can "average down" on an existing position if its price has dropped by a specified percentage (e.g., 3%) from the last purchase price. It prioritizes the one with the biggest current drop.

### Exit Rules (checked daily at 3:20 PM)

- **Check for Gains:** The script scans the portfolio for any stock trading above a specified profit target (e.g., 5%) from its average purchase price.
- **Sell for Profit:** If multiple stocks meet the profit criteria, the script sells only one per day—the one with the highest percentage gain.

### Capital Allocation

The script supports two sizing models:

- **Fixed Amount:** A constant, predefined amount is invested in each trade.
- **Dynamic Amount:** A percentage of the total current portfolio value is invested in each trade (e.g., 1/40th of the portfolio).

---

## How to Use

### 1. Installation

Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Before running the backtest, you can customize the strategy parameters in the `nifty_shop_backtest_v4.py` script under the **MASTER CONFIGURATION & STRATEGY RULES** section.

Key parameters you can change:

- `START_DATE` / `END_DATE`: The period for the backtest.
- `INITIAL_CAPITAL`: The starting amount for the simulation.
- `PROFIT_TARGET_PERCENT`: The gain required to trigger a sell.
- `AVERAGING_TRIGGER_PERCENT`: The loss required to trigger an average-down buy.
- `MAX_NEW_BUYS_PER_DAY`: Limits how many new stocks can be bought in a single day.
- `TRADE_AMOUNT_FIXED` / `DYNAMIC_SIZING_DIVISOR`: Controls the trade size for the two models.
- `ENABLE_SIP`: Set to `True` to simulate regular monthly investments.
- `RUN_FIXED_SIZING_TEST` / `RUN_DYNAMIC_SIZING_TEST`: Booleans to control which backtesting models to run.

### 3. Execution

Run the backtest from your terminal:

```bash
python nifty_shop_backtest_v4.py
```

---

## Understanding the Backtest Script

The `nifty_shop_backtest_v4.py` script is organized into four main parts:

1.  **Configuration & Data Fetching:**

    - Sets all strategy rules, dates, and capital settings.
    - Loads historical Nifty 50 constituents from `nifty50.csv` to ensure the backtest uses the correct stocks for any given date.
    - Downloads all required historical price data from Yahoo Finance using the `yfinance` library.

2.  **Backtesting Engine (`run_backtest` function):**

    - This is the core of the simulation. It iterates through each day in the specified date range.
    - On each day, it processes SIP injections, checks for sell signals based on profit targets, and then checks for buy signals based on the 20 SMA deviation.
    - It meticulously logs every transaction (buy, sell, average, SIP) and calculates the portfolio's total value daily.
    - It handles portfolio management, including cash, holdings, and calculating average costs.

3.  **Metrics Calculation & Reporting:**

    - After the simulation loop, this section calculates a wide array of performance metrics.
    - **Returns Analysis:** CAGR (Compounded Annual Growth Rate) and XIRR (for cash-flow-adjusted returns).
    - **Risk-Adjusted Returns:** Sharpe Ratio (returns vs. volatility) and Sortino Ratio (returns vs. downside volatility).
    - **Risk Metrics:** Maximum Drawdown (largest peak-to-trough drop) and Calmar Ratio (CAGR vs. Max Drawdown).
    - **Market Comparison:** Alpha (outperformance vs. benchmark) and Beta (volatility relative to the benchmark).
    - The results are printed to the console in a structured format.

4.  **Plotting:**
    - Generates a comparative equity curve chart using `matplotlib`.
    - This visualizes the growth of the strategy's portfolio value against the NIFTYBEES index, providing an intuitive understanding of its performance over time.

---

## Backtest Output Explained

The script provides a rich set of outputs to help you analyze the strategy's performance:

- **Console Report:**
  - **Detailed Strategy Metrics:** The main table showing CAGR, XIRR, Alpha, Beta, Sharpe Ratio, etc.
  - **Quarterly Trade Grid:** A pivot table showing the number of buys and sells in each quarter, helping to identify periods of high/low activity.
  - **Top Trades Analysis:** Lists of the top 5 trades based on the number of averaging events and profitability (both by percentage and absolute amount).
  - **Recent Activity:** The last 10 transactions and the last 10 closed trades are displayed for a recent snapshot.
  - **Final Holdings:** A summary of all stocks held in the portfolio at the end of the backtest period.
- **CSV File:**
  - `trade_log.csv`: A detailed CSV file is generated, containing a timestamped record of every single transaction made during the backtest. This is useful for in-depth, external analysis.
- **Chart:**
  - **Equity Curve Plot:** A PNG window will pop up showing the strategy's performance graph.

---

## Credits

This strategy was inspired by the "Nifty Shop" concept by **fab trader**. You can watch the original video here: [https://www.youtube.com/watch?v=GQ3gjYaislM](https://www.youtube.com/watch?v=GQ3gjYaislM)

## Disclaimer

This project is for educational and informational purposes only. The trading strategy and backtest results are not financial advice. Trading in the stock market involves risk, and you should consult with a qualified financial advisor before making any investment decisions. The author and contributors are not responsible for any financial losses incurred.
