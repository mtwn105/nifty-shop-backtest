import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import numpy_financial as npf
import time
from pyxirr import xirr


def get_nifty50_tickers():
    """
    Reads the nifty50.csv file and returns a list of stock tickers.
    """
    df = pd.read_csv("nifty50.csv")
    return df["Symbol"].tolist()


def download_data(tickers, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance in chunks with retries.
    """
    all_data = {}
    chunk_size = 5
    retries = 3
    timeout = 60  # seconds

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        for attempt in range(retries):
            try:
                print(f"Downloading {chunk} (Attempt {attempt + 1}/{retries})...")
                data = yf.download(
                    chunk,
                    start=start_date,
                    end=end_date,
                    group_by="ticker",
                    timeout=timeout,
                )
                all_data.update(data.to_dict())
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Could not download {chunk}: {e}")
                if attempt < retries - 1:
                    print("Retrying...")
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    print(f"Failed to download {chunk} after {retries} attempts.")

    # Reconstruct the DataFrame
    df = pd.DataFrame.from_dict(all_data)
    # The columns are now tuples of (ticker, attribute), which is what the rest of the script expects.
    # We need to ensure the index is a DatetimeIndex
    df.index = pd.to_datetime(df.index)
    return df


def backtest_strategy(data, tickers, initial_capital, investment_per_trade):
    """
    Backtests the Nifty Shop strategy.
    """
    capital = initial_capital
    portfolio = {}
    trade_log = []
    daily_portfolio_values = []

    # Prepare data
    for ticker in tickers:
        if (ticker, "Close") in data.columns:
            data[(ticker, "20_SMA")] = data[(ticker, "Close")].rolling(window=20).mean()

    for date in data.index:
        # Calculate daily portfolio value at the start of the day
        current_holdings_value = 0
        for stock, trades in portfolio.items():
            if trades:
                total_shares = sum(t["shares"] for t in trades)
                price = data.loc[date, (stock, "Close")]
                if not pd.isna(price):
                    current_holdings_value += total_shares * price
        daily_portfolio_values.append(capital + current_holdings_value)

        # Exit Logic (3:20 PM)
        if not portfolio:
            pass
        else:
            gains = []
            for stock, trades in portfolio.items():
                if not trades:
                    continue
                avg_price = np.mean([t["price"] for t in trades])
                current_price = data.loc[date, (stock, "Close")]
                if pd.isna(current_price):
                    continue
                gain = (current_price - avg_price) / avg_price
                if gain > 0.05:
                    gains.append({"stock": stock, "gain": gain, "price": current_price})

            if gains:
                stock_to_sell = max(gains, key=lambda x: x["gain"])
                stock = stock_to_sell["stock"]
                sell_price = stock_to_sell["price"]

                total_shares = sum(t["shares"] for t in portfolio[stock])
                capital += total_shares * sell_price

                trade_log.append(
                    {
                        "date": date,
                        "stock": stock,
                        "action": "SELL",
                        "price": sell_price,
                        "shares": total_shares,
                        "capital": capital,
                    }
                )
                portfolio[stock] = []

        # Entry Logic (End of Day)
        potential_buys = []
        for ticker in tickers:
            if (ticker, "Close") in data.columns and (ticker, "20_SMA") in data.columns:
                current_price = data.loc[date, (ticker, "Close")]
                sma_20 = data.loc[date, (ticker, "20_SMA")]
                if pd.isna(current_price) or pd.isna(sma_20):
                    continue

                if current_price < sma_20:
                    distance = (sma_20 - current_price) / sma_20
                    potential_buys.append(
                        {"ticker": ticker, "distance": distance, "price": current_price}
                    )

        if not potential_buys:
            continue

        top_5_dips = sorted(potential_buys, key=lambda x: x["distance"], reverse=True)[
            :5
        ]

        # Averaging Down Logic
        owned_stocks_in_dips = [
            s for s in top_5_dips if s["ticker"] in portfolio and portfolio[s["ticker"]]
        ]
        if len(owned_stocks_in_dips) == 5:
            stocks_to_average = []
            for stock_info in owned_stocks_in_dips:
                ticker = stock_info["ticker"]
                last_purchase_price = portfolio[ticker][-1]["price"]
                current_price = stock_info["price"]
                if (last_purchase_price - current_price) / last_purchase_price > 0.03:
                    drop = (last_purchase_price - current_price) / last_purchase_price
                    stocks_to_average.append(
                        {"ticker": ticker, "drop": drop, "price": current_price}
                    )

            if stocks_to_average:
                stock_to_average = max(stocks_to_average, key=lambda x: x["drop"])
                ticker = stock_to_average["ticker"]
                price = stock_to_average["price"]
                if capital >= investment_per_trade:
                    shares = int(investment_per_trade / price)
                    if shares == 0:
                        continue
                    capital -= shares * price
                    portfolio[ticker].append({"price": price, "shares": shares})
                    trade_log.append(
                        {
                            "date": date,
                            "stock": ticker,
                            "action": "BUY (Avg Down)",
                            "price": price,
                            "shares": shares,
                            "capital": capital,
                        }
                    )
            continue  # Skip to next day after averaging down

        # Buy New Positions Logic
        buy_count = 0
        for stock_info in top_5_dips:
            if buy_count >= 2:
                break

            ticker = stock_info["ticker"]
            if ticker not in portfolio or not portfolio[ticker]:
                if capital >= investment_per_trade:
                    price = stock_info["price"]
                    shares = int(investment_per_trade / price)
                    if shares == 0:
                        continue
                    capital -= shares * price
                    if ticker not in portfolio:
                        portfolio[ticker] = []
                    portfolio[ticker].append({"price": price, "shares": shares})
                    trade_log.append(
                        {
                            "date": date,
                            "stock": ticker,
                            "action": "BUY",
                            "price": price,
                            "shares": shares,
                            "capital": capital,
                        }
                    )
                    buy_count += 1

    return (
        trade_log,
        capital,
        portfolio,
        pd.Series(daily_portfolio_values, index=data.index),
    )


def plot_results(equity_curve, benchmark_data, initial_capital):
    """
    Plots the equity curve and drawdown against a benchmark.
    """
    plt.figure(figsize=(12, 8))

    # Equity Curve vs Benchmark
    plt.subplot(2, 1, 1)

    # Normalize strategy equity curve to show returns
    strategy_returns = (equity_curve / initial_capital) - 1
    plt.plot(strategy_returns * 100, label="Strategy")

    # Normalize benchmark data to show returns
    if not benchmark_data.empty:
        benchmark_returns = (
            benchmark_data["Close"] / benchmark_data["Close"].iloc[0]
        ) - 1
        plt.plot(benchmark_returns * 100, label="NIFTYBEES", linestyle="--")

    plt.title("Strategy vs. NIFTYBEES - Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()

    # Drawdown
    plt.subplot(2, 1, 2)
    drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
    plt.plot(drawdown, color="red")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png")
    print("\nPlots saved to backtest_results.png")


def calculate_xirr(
    trade_log, initial_capital, final_portfolio_value, start_date, end_date
):
    """
    Calculates the XIRR of the strategy.
    """
    if not trade_log:
        return 0

    dates = [pd.to_datetime(start_date)]
    values = [-initial_capital]

    for trade in trade_log:
        dates.append(trade["date"])
        if "BUY" in trade["action"]:
            values.append(-trade["price"] * trade["shares"])
        else:  # SELL
            values.append(trade["price"] * trade["shares"])

    dates.append(pd.to_datetime(end_date))
    values.append(final_portfolio_value)

    try:
        # Use pyxirr for accurate XIRR calculation
        return xirr(np.array(dates), np.array(values)) * 100
    except Exception as e:
        print(f"Could not calculate XIRR: {e}")
        return 0


def analyze_results(
    trade_log,
    final_capital,
    initial_capital,
    portfolio,
    data,
    benchmark_data,
    end_date,
    equity_curve,
):
    """
    Analyzes and prints the backtest results.
    """
    if not trade_log:
        print("No trades were made during the backtest period.")
        return

    log_df = pd.DataFrame(trade_log)

    print("\n--- Backtest Results ---")
    print(f"Initial Capital: {initial_capital:,.2f}")

    # Calculate final portfolio value
    final_portfolio_value = final_capital
    for stock, trades in portfolio.items():
        if trades:
            total_shares = sum(t["shares"] for t in trades)
            last_price = data.loc[data.index[-1], (stock, "Close")]
            if not pd.isna(last_price):
                final_portfolio_value += total_shares * last_price

    print(f"Final Portfolio Value: {final_portfolio_value:,.2f}")
    total_return = (final_portfolio_value - initial_capital) / initial_capital * 100
    print(f"Total Return: {total_return:.2f}%")

    plot_results(equity_curve, benchmark_data, initial_capital)

    # Calculate and print XIRR
    xirr_val = calculate_xirr(
        trade_log, initial_capital, final_portfolio_value, START_DATE, END_DATE
    )
    print(f"Annualized XIRR: {xirr_val:.2f}%")

    sells = log_df[log_df["action"] == "SELL"]
    buys = log_df[log_df["action"].str.contains("BUY")]

    print(f"\nTotal Trades: {len(buys)}")
    print(f"Winning Trades: {len(sells)}")

    if len(sells) > 0:
        win_rate = len(sells) / len(buys) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    else:
        print("Win Rate: 0.00%")


if __name__ == "__main__":
    # Parameters
    START_DATE = "2020-01-01"
    END_DATE = "2025-07-25"
    INITIAL_CAPITAL = 100000  # 10 Lakhs
    INVESTMENT_PER_TRADE = 3000

    # Run Backtest
    nifty_tickers = get_nifty50_tickers()
    # Append '.NS' for Yahoo Finance compatibility
    tickers_ns = [ticker + ".NS" for ticker in nifty_tickers]

    print("Downloading data...")
    stock_data = download_data(tickers_ns, START_DATE, END_DATE)

    print("Downloading benchmark data...")
    try:
        benchmark_data = yf.download(
            "NIFTYBEES.NS", start=START_DATE, end=END_DATE, timeout=60
        )
    except Exception as e:
        print(f"Could not download benchmark data: {e}")
        benchmark_data = pd.DataFrame()  # Empty dataframe to avoid errors later

    print("Running backtest...")
    (
        trade_log,
        final_capital,
        final_portfolio,
        equity_curve,
    ) = backtest_strategy(stock_data, tickers_ns, INITIAL_CAPITAL, INVESTMENT_PER_TRADE)

    # Analyze and Print Results
    analyze_results(
        trade_log,
        final_capital,
        INITIAL_CAPITAL,
        final_portfolio,
        stock_data,
        benchmark_data,
        END_DATE,
        equity_curve,
    )

    # Save trade log to CSV
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        log_df.to_csv("trade_log.csv", index=False)
        print("\nTrade log saved to trade_log.csv")
