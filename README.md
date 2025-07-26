# Nifty Shop Backtesting

This project is for backtesting the "Nifty Shop" trading strategy on the Nifty 50 index.

## Strategy: Nifty Shop

The "Nifty Shop" strategy is designed for beginners and focuses on Nifty50 stocks.

### Entry Rules (checked daily at 3:20 PM)

- **Scan for Dips:** Identify the top five Nifty50 stocks that are trading the furthest below their 20-day moving average.
- **Buy New Positions:** From the list of five, you can buy up to two stocks that you don't already own.
- **Averaging Down:** If you already own all five of the identified stocks, you can "average down" on one of your existing holdings. This means buying more of a stock that has dropped more than 3% from your last purchase price, prioritizing the one with the biggest drop.

### Exit Rules (checked daily at 3:20 PM)

- **Check for Gains:** Look for any stock in your portfolio that is trading more than 5% above your average purchase price.
- **Sell for Profit:** Sell only one stock per day, choosing the one that has the highest gain.

### Capital Allocation

- **Fixed Amount:** You can choose to invest a fixed amount, for example, 15,000 per trade.
- **Dynamic Amount:** Alternatively, you can divide your total trading capital by 40 and use that amount for each trade. This allows your investment size to grow as your capital grows.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the backtest, execute the `nifty_shop_backtest_v4.py` script:

```bash
python nifty_shop_backtest_v4.py
```

You can configure the backtest by editing the "MASTER CONFIGURATION & STRATEGY RULES" section in the script. You can enable or disable fixed and dynamic sizing tests using `RUN_FIXED_SIZING_TEST` and `RUN_DYNAMIC_SIZING_TEST` variables.

## Backtest Output

The script will output the following:

- **Detailed Strategy Metrics:** A comprehensive report in the console including CAGR, XIRR, Alpha, Beta, Sharpe Ratio, Sortino Ratio, Max Drawdown, and more.
- **Quarterly Trade Grid:** A summary of buy and sell transactions per quarter.
- **Top Trades:** Lists of top 5 trades by averaging events and profitability (both percentage and amount).
- **Transaction Logs:** The last 10 transactions and last 10 closed trades are displayed. A full `trade_log.csv` is also generated.
- **Final Holdings:** A summary of the portfolio at the end of the backtest period.
- **Equity Curve Plot:** A chart comparing the performance of the strategy (with fixed and/or dynamic sizing) against the NIFTYBEES benchmark.

## Credits

This strategy was inspired by the "Nifty Shop" concept by **fab trader**. You can watch the original video here: [https://www.youtube.com/watch?v=GQ3gjYaislM](https://www.youtube.com/watch?v=GQ3gjYaislM)

## Disclaimer

This project is for educational and informational purposes only. The trading strategy and backtest results are not financial advice. Trading in the stock market involves risk, and you should consult with a qualified financial advisor before making any investment decisions. The author and contributors are not responsible for any financial losses incurred.
