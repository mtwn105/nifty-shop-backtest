import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from queue import Queue
from threading import Thread


def xirr(dates, values, guess=0.1):
    """
    Calculate the Internal Rate of Return (XIRR) using a manual Newton-Raphson implementation.
    """
    if not values or len(dates) != len(values):
        return None

    # Ensure transactions are sorted by date
    transactions = sorted(zip(dates, values))
    dates, values = zip(*transactions)

    # Use 365.25 to account for leap years, which is more accurate
    years = [(d - dates[0]).days / 365.25 for d in dates]

    def npv(rate):
        return sum(v / ((1 + rate) ** y) for v, y in zip(values, years))

    def d_npv(rate):
        # Derivative of the NPV function with respect to the rate
        return sum(-y * v / ((1 + rate) ** (y + 1)) for v, y in zip(values, years))

    # --- Manual Newton-Raphson Implementation ---
    rate = guess
    for _ in range(100):  # Max 100 iterations
        npv_val = npv(rate)
        d_npv_val = d_npv(rate)

        if abs(npv_val) < 1e-6:  # Convergence check
            return rate

        if d_npv_val == 0:  # Avoid division by zero
            return None

        rate = rate - npv_val / d_npv_val

    return None  # Return None if no convergence


# --- 1. MASTER CONFIGURATION & STRATEGY RULES ---

# -- General Settings --
START_DATE = "2020-01-01"
END_DATE = "2025-07-25"
INITIAL_CAPITAL = 1000000.00
FLAT_BROKERAGE_FEE = 20.00
RISK_FREE_RATE_ANNUAL = 0.07  # Example: 7% for risk-adjusted return calculations

# -- Strategy Rules --
PROFIT_TARGET_PERCENT = 5
AVERAGING_TRIGGER_PERCENT = 3
MAX_NEW_BUYS_PER_DAY = 1
TRADE_AMOUNT_FIXED = 3000.00
DYNAMIC_SIZING_DIVISOR = 40
MAX_AVERAGING_EVENTS = None  # Set to None for no limit
SELL_IF_REMOVED_FROM_NIFTY50 = False

# -- SIP Settings --
ENABLE_SIP = False
SIP_AMOUNT = 10000.00
SIP_DAY = 1

# -- Dividend Settings --
REINVEST_DIVIDENDS = True

# -- Control Which Backtests to Run --
RUN_FIXED_SIZING_TEST = False
RUN_DYNAMIC_SIZING_TEST = True

# --- (No changes needed below this line) ---


# Tickers and Data Fetching
def get_nifty50_tickers_for_date(date, nifty_composition_df):
    """Gets the list of Nifty 50 tickers for a specific date."""
    month_year_str = date.strftime("%b-%y")
    if month_year_str in nifty_composition_df.columns:
        return (nifty_composition_df[month_year_str].dropna() + ".NS").tolist()
    # Fallback to the most recent composition if the date is out of bounds
    return (nifty_composition_df.iloc[:, -1].dropna() + ".NS").tolist()


try:
    nifty_composition_df = pd.read_csv("nifty50.csv")
    nifty_composition_df.columns = nifty_composition_df.columns.str.strip()
    # Get a set of all unique tickers that have ever been in the Nifty 50
    all_historical_tickers = set()
    for col in nifty_composition_df.columns:
        all_historical_tickers.update(nifty_composition_df[col].dropna().tolist())
    NIFTY50_TICKERS_ALL_TIME = [ticker + ".NS" for ticker in all_historical_tickers]
    print("Successfully loaded historical Nifty 50 compositions from nifty50.csv")
except Exception as e:
    print(f"Could not read or process nifty50.csv: {e}. Backtest cannot proceed.")
    exit()

NIFTY_INDEX_TICKER = "NIFTYBEES.NS"  # Use NIFTYBEES as benchmark

print("Fetching all required historical data... (This may take a minute)")
data_start_date = (
    datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=40)
).strftime("%Y-%m-%d")
all_tickers_to_fetch = NIFTY50_TICKERS_ALL_TIME + [NIFTY_INDEX_TICKER]
all_data = yf.download(
    all_tickers_to_fetch, start=data_start_date, end=END_DATE, auto_adjust=True
)["Close"]
nifty_data = all_data[NIFTY_INDEX_TICKER].dropna()
stock_data = all_data.drop(columns=[NIFTY_INDEX_TICKER], errors="ignore")

# Ensure the index is unique to prevent errors during lookup
if not stock_data.index.is_unique:
    print("Warning: Duplicate dates found in historical data. Consolidating...")
    stock_data = stock_data.loc[~stock_data.index.duplicated(keep="last")]

sma_20 = stock_data.rolling(window=20).mean()
print("Data fetching complete.")


# --- Optimized Dividend Data Fetching ---
def fetch_single_dividend(ticker, q):
    """Fetches dividend data for a single ticker and puts it in a queue."""
    try:
        stock_obj = yf.Ticker(ticker)
        dividends = stock_obj.dividends
        if not dividends.empty:
            dividends_in_range = dividends.loc[START_DATE:END_DATE]
            if not dividends_in_range.empty:
                dividends_in_range.index = dividends_in_range.index.tz_localize(None)
                q.put({ticker: dividends_in_range})
    except Exception:
        pass  # Ignore tickers that fail


def get_all_dividends_threaded(tickers):
    """Fetches dividend data for a list of tickers using multiple threads."""
    print("Fetching dividend data (threaded)...")
    q = Queue()
    threads = []
    for ticker in tickers:
        thread = Thread(target=fetch_single_dividend, args=(ticker, q))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()  # Wait for all threads to finish

    all_dividends = {}
    while not q.empty():
        all_dividends.update(q.get())

    print("Dividend data fetching complete.")
    return all_dividends


all_dividends = get_all_dividends_threaded(all_tickers_to_fetch)


# --- 2. BACKTESTING FUNCTION (MODULARIZED) ---
def run_backtest(sizing_method):
    print(f"\n{'='*40}\n RUNNING BACKTEST (SIZING METHOD: {sizing_method}) \n{'='*40}")

    portfolio = {"cash": INITIAL_CAPITAL, "holdings": {}, "value_history": []}
    transactions_list = []
    closed_trades_log = []
    cash_flows = []  # Initialize empty cash flows for XIRR
    non_reinvested_dividends = 0  # Track dividends separately if not reinvested

    # Find the first actual trading day to start the backtest
    first_trading_day = stock_data.loc[
        stock_data.index >= pd.to_datetime(START_DATE)
    ].index.min()

    # Adjust backtest range to start from the first trading day
    backtest_range = pd.date_range(start=first_trading_day, end=END_DATE)
    profit_target_multiplier = 1 + (PROFIT_TARGET_PERCENT / 100)
    averaging_trigger_multiplier = 1 - (AVERAGING_TRIGGER_PERCENT / 100)
    last_sip_month = None

    # Add initial capital to cash flows for XIRR calculation
    cash_flows.append((first_trading_day, -INITIAL_CAPITAL))

    for current_date in backtest_range:
        if current_date not in stock_data.index:
            continue

        # --- SIP Injection Logic ---
        if ENABLE_SIP and current_date.day >= SIP_DAY:
            current_month_year = (current_date.year, current_date.month)
            if current_month_year != last_sip_month:
                portfolio["cash"] += SIP_AMOUNT
                cash_flows.append((current_date, -SIP_AMOUNT))
                last_sip_month = current_month_year
                transactions_list.append(
                    {
                        "Date": current_date,
                        "Ticker": "SIP",
                        "Type": "DEPOSIT",
                        "Price": SIP_AMOUNT,
                        "Quantity": 1,
                        "PnL_%": 0,
                        "Capital": portfolio["cash"],
                        "Reason": "Monthly SIP",
                    }
                )

        # --- Dividend Income Logic ---
        # Check for dividends on the current date for all held stocks
        for ticker, lots in portfolio["holdings"].items():
            if ticker in all_dividends and current_date in all_dividends[ticker].index:
                dividend_per_share = all_dividends[ticker].loc[current_date]
                total_quantity = sum(lot["quantity"] for lot in lots)

                if total_quantity > 0 and dividend_per_share > 0:
                    total_dividend_received = dividend_per_share * total_quantity
                    if REINVEST_DIVIDENDS:
                        portfolio["cash"] += total_dividend_received
                    else:
                        non_reinvested_dividends += total_dividend_received

                    transactions_list.append(
                        {
                            "Date": current_date,
                            "Ticker": ticker,
                            "Type": "DIVIDEND",
                            "Price": dividend_per_share,
                            "Quantity": total_quantity,
                            "PnL_%": 0,  # Not applicable for dividends
                            "Capital": portfolio["cash"],
                            "Reason": f"Dividend Received (₹{dividend_per_share:.2f}/share)",
                        }
                    )

        current_holdings_value = sum(
            stock_data.loc[current_date, ticker] * lot["quantity"]
            for ticker, lots in portfolio["holdings"].items()
            for lot in lots
            if ticker in stock_data.columns
            and pd.notna(stock_data.loc[current_date, ticker])
        )
        current_portfolio_value = portfolio["cash"] + current_holdings_value
        trade_amount = (
            TRADE_AMOUNT_FIXED
            if sizing_method == "FIXED"
            else current_portfolio_value / DYNAMIC_SIZING_DIVISOR
        )

        NIFTY50_TICKERS_CURRENT = get_nifty50_tickers_for_date(
            current_date, nifty_composition_df
        )

        # Sell stocks no longer in Nifty 50
        if SELL_IF_REMOVED_FROM_NIFTY50:
            stocks_to_sell_delisted = [
                ticker
                for ticker in portfolio["holdings"]
                if ticker not in NIFTY50_TICKERS_CURRENT
            ]
            for ticker_to_sell in stocks_to_sell_delisted:
                if ticker_to_sell in stock_data.columns and pd.notna(
                    stock_data.loc[current_date, ticker_to_sell]
                ):
                    lots = portfolio["holdings"][ticker_to_sell]
                    total_qty_sold = sum(lot["quantity"] for lot in lots)
                    sell_price = stock_data.loc[current_date, ticker_to_sell]
                    sell_value = total_qty_sold * sell_price
                    avg_cost_to_log = (
                        sum(l["quantity"] * l["price"] for l in lots) / total_qty_sold
                    )
                    pnl_percent = (
                        ((sell_price - avg_cost_to_log) / avg_cost_to_log) * 100
                        if avg_cost_to_log > 0
                        else 0
                    )
                    portfolio["cash"] += sell_value - FLAT_BROKERAGE_FEE
                    transactions_list.append(
                        {
                            "Date": current_date,
                            "Ticker": ticker_to_sell,
                            "Type": "SELL",
                            "Price": sell_price,
                            "Quantity": total_qty_sold,
                            "PnL_%": pnl_percent,
                            "Capital": portfolio["cash"],
                            "Reason": "Removed from Nifty 50",
                        }
                    )
                    closed_trades_log.append(
                        {
                            "Ticker": ticker_to_sell,
                            "EntryDate": lots[0]["date"],
                            "ExitDate": current_date,
                            "AvgCost": avg_cost_to_log,
                            "ExitPrice": sell_price,
                            "Quantity": total_qty_sold,
                            "AveragingEvents": sum(
                                1 for lot in lots if lot["type"] == "AVG"
                            ),
                        }
                    )
                    del portfolio["holdings"][ticker_to_sell]

        # Sell Logic
        eligible_for_sell = []
        for ticker, lots in portfolio["holdings"].items():
            if ticker in stock_data.columns and pd.notna(
                stock_data.loc[current_date, ticker]
            ):
                total_qty = sum(lot["quantity"] for lot in lots)
                avg_price = (
                    sum(lot["quantity"] * lot["price"] for lot in lots) / total_qty
                )
                if (
                    stock_data.loc[current_date, ticker]
                    >= avg_price * profit_target_multiplier
                ):
                    gain = (
                        stock_data.loc[current_date, ticker] - avg_price
                    ) / avg_price
                    eligible_for_sell.append((ticker, gain, avg_price))
        if eligible_for_sell:
            eligible_for_sell.sort(key=lambda x: x[1], reverse=True)
            ticker_to_sell, _, avg_cost_to_log = eligible_for_sell[0]
            lots = portfolio["holdings"][ticker_to_sell]
            total_qty_sold = sum(lot["quantity"] for lot in lots)
            sell_price = stock_data.loc[current_date, ticker_to_sell]
            sell_value = total_qty_sold * sell_price
            pnl_percent = (
                ((sell_price - avg_cost_to_log) / avg_cost_to_log) * 100
                if avg_cost_to_log > 0
                else 0
            )
            portfolio["cash"] += sell_value - FLAT_BROKERAGE_FEE
            transactions_list.append(
                {
                    "Date": current_date,
                    "Ticker": ticker_to_sell,
                    "Type": "SELL",
                    "Price": sell_price,
                    "Quantity": total_qty_sold,
                    "PnL_%": pnl_percent,
                    "Capital": portfolio["cash"],
                    "Reason": "Profit target hit",
                }
            )

            closed_trades_log.append(
                {
                    "Ticker": ticker_to_sell,
                    "EntryDate": lots[0]["date"],
                    "ExitDate": current_date,
                    "AvgCost": avg_cost_to_log,
                    "ExitPrice": sell_price,
                    "Quantity": total_qty_sold,
                    "AveragingEvents": sum(1 for lot in lots if lot["type"] == "AVG"),
                }
            )
            del portfolio["holdings"][ticker_to_sell]

        # Buy Logic
        deviations = []
        for ticker in NIFTY50_TICKERS_CURRENT:
            if (
                ticker in stock_data.columns
                and pd.notna(stock_data.loc[current_date, ticker])
                and pd.notna(sma_20.loc[current_date, ticker])
            ):
                if (
                    stock_data.loc[current_date, ticker]
                    < sma_20.loc[current_date, ticker]
                ):
                    deviations.append(
                        (
                            ticker,
                            (
                                stock_data.loc[current_date, ticker]
                                - sma_20.loc[current_date, ticker]
                            )
                            / sma_20.loc[current_date, ticker],
                        )
                    )
        deviations.sort(key=lambda x: x[1])
        top_5_fallers = [ticker for ticker, _ in deviations[:5]]
        all_in_portfolio = (
            all(ticker in portfolio["holdings"] for ticker in top_5_fallers)
            if top_5_fallers
            else True
        )

        if not all_in_portfolio and top_5_fallers:
            new_buys_today = 0
            for ticker in top_5_fallers:
                if new_buys_today >= MAX_NEW_BUYS_PER_DAY:
                    break
                if ticker not in portfolio["holdings"]:
                    buy_price = stock_data.loc[current_date, ticker]
                    if buy_price > 0:
                        quantity = int(trade_amount // buy_price)
                        buy_value = quantity * buy_price
                        if (
                            quantity > 0
                            and portfolio["cash"] >= buy_value + FLAT_BROKERAGE_FEE
                        ):
                            portfolio["cash"] -= buy_value + FLAT_BROKERAGE_FEE
                            portfolio["holdings"][ticker] = [
                                {
                                    "date": current_date,
                                    "quantity": quantity,
                                    "price": buy_price,
                                    "type": "BUY",
                                }
                            ]
                            transactions_list.append(
                                {
                                    "Date": current_date,
                                    "Ticker": ticker,
                                    "Type": "BUY",
                                    "Price": buy_price,
                                    "Quantity": quantity,
                                    "PnL_%": 0,
                                    "Capital": portfolio["cash"],
                                    "Reason": "Top faller below 20 SMA",
                                }
                            )
                            new_buys_today += 1
        elif top_5_fallers:
            eligible_for_avg = []
            for ticker, lots in portfolio["holdings"].items():
                if ticker in stock_data.columns and pd.notna(
                    stock_data.loc[current_date, ticker]
                ):
                    avg_price = sum(l["quantity"] * l["price"] for l in lots) / sum(
                        l["quantity"] for l in lots
                    )
                    current_averages = sum(1 for lot in lots if lot["type"] == "AVG")
                    if (
                        MAX_AVERAGING_EVENTS is None
                        or current_averages < MAX_AVERAGING_EVENTS
                    ):
                        if (
                            stock_data.loc[current_date, ticker]
                            <= avg_price * averaging_trigger_multiplier
                        ):
                            eligible_for_avg.append(
                                (
                                    ticker,
                                    (stock_data.loc[current_date, ticker] - avg_price)
                                    / avg_price,
                                )
                            )
            if eligible_for_avg:
                eligible_for_avg.sort(key=lambda x: x[1])
                ticker_to_avg, _ = eligible_for_avg[0]
                buy_price = stock_data.loc[current_date, ticker_to_avg]
                if buy_price > 0:
                    quantity_new = int(trade_amount // buy_price)
                    buy_value = quantity_new * buy_price
                    if (
                        quantity_new > 0
                        and portfolio["cash"] >= buy_value + FLAT_BROKERAGE_FEE
                    ):
                        portfolio["cash"] -= buy_value + FLAT_BROKERAGE_FEE
                        portfolio["holdings"][ticker_to_avg].append(
                            {
                                "date": current_date,
                                "quantity": quantity_new,
                                "price": buy_price,
                                "type": "AVG",
                            }
                        )
                        transactions_list.append(
                            {
                                "Date": current_date,
                                "Ticker": ticker_to_avg,
                                "Type": "AVG",
                                "Price": buy_price,
                                "Quantity": quantity_new,
                                "PnL_%": 0,
                                "Capital": portfolio["cash"],
                                "Reason": "Averaging trigger hit",
                            }
                        )

        final_holdings_value = sum(
            stock_data.loc[current_date, ticker] * lot["quantity"]
            for ticker, lots in portfolio["holdings"].items()
            for lot in lots
            if ticker in stock_data.columns
            and pd.notna(stock_data.loc[current_date, ticker])
        )
        portfolio["value_history"].append(
            {
                "Date": current_date,
                "Value": portfolio["cash"] + final_holdings_value,
                "Deployed Capital": final_holdings_value,
            }
        )

    # --- Reporting for this run ---
    pf_value_df = pd.DataFrame(portfolio["value_history"]).set_index("Date")
    transactions_log_df = pd.DataFrame(transactions_list)  # For unified reporting

    print("\n--- Strategy Assumptions ---")
    if sizing_method == "FIXED":
        print(f"Sizing Model: FIXED (₹{TRADE_AMOUNT_FIXED:,.2f} per trade)")
    else:
        print(
            f"Sizing Model: DYNAMIC (1/{DYNAMIC_SIZING_DIVISOR} of portfolio value per trade)"
        )
    print(f"Strategy Budget: {INITIAL_CAPITAL:,.2f}")
    print(
        f"Profit Target: {PROFIT_TARGET_PERCENT:.1f}% | Averaging Trigger: {AVERAGING_TRIGGER_PERCENT:.1f}% Drop"
    )
    print(f"Max New Buys Per Day: {MAX_NEW_BUYS_PER_DAY}")
    max_avg_text = (
        "Unlimited" if MAX_AVERAGING_EVENTS is None else str(MAX_AVERAGING_EVENTS)
    )
    print(f"Max Averaging Allowed: {max_avg_text}")
    if ENABLE_SIP:
        print(f"SIP Enabled: Yes (₹{SIP_AMOUNT:,.2f} on day {SIP_DAY} of each month)")
    else:
        print("SIP Enabled: No")
    print(f"Reinvest Dividends: {'Yes' if REINVEST_DIVIDENDS else 'No'}")

    if (
        not pf_value_df.empty
        and (pf_value_df.index[-1] - pf_value_df.index[0]).days > 0
    ):
        years = (pf_value_df.index[-1] - pf_value_df.index[0]).days / 365.25
        final_portfolio_value = pf_value_df["Value"].iloc[-1]
        if not REINVEST_DIVIDENDS:
            final_portfolio_value += non_reinvested_dividends

        # <<< FIX IMPLEMENTED HERE >>>
        # Calculate daily returns for the strategy
        daily_returns = pf_value_df["Value"].pct_change()

        # Align NIFTY data to the portfolio's calendar-day index, forward-filling weekends/holidays
        nifty_aligned = nifty_data.reindex(pf_value_df.index, method="ffill")
        nifty_daily_returns = nifty_aligned.pct_change()

        # Align the two return series to ensure we only compare days where both exist
        aligned_strat, aligned_nifty = daily_returns.align(
            nifty_daily_returns, join="inner"
        )
        aligned_strat.dropna(inplace=True)
        aligned_nifty.dropna(inplace=True)
        aligned_strat, aligned_nifty = aligned_strat.align(aligned_nifty, join="inner")
        # <<< END OF FIX >>>

        cagr_strategy = (
            (final_portfolio_value / INITIAL_CAPITAL) ** (1 / years) - 1
        ) * 100

        # --- XIRR Calculation ---
        # Get the market value of the final holdings, which represents the final cash inflow
        final_holdings_value = pf_value_df["Deployed Capital"].iloc[-1]
        cash_flows.append((pf_value_df.index[-1], final_holdings_value))

        # Aggregate cash flows by date, which is a requirement for some XIRR implementations
        # and a good practice to ensure consistency.
        from collections import defaultdict

        aggregated_cash_flows = defaultdict(float)
        for date, value in cash_flows:
            aggregated_cash_flows[date] += value

        # Convert the aggregated dictionary back to sorted lists
        sorted_cash_flows = sorted(aggregated_cash_flows.items())
        if not sorted_cash_flows:
            strategy_xirr = None
        else:
            dates, values = zip(*sorted_cash_flows)
            strategy_xirr = xirr(dates, values)

        # For CAGR, use the original (un-daily) nifty data over the same date range
        nifty_date_range = nifty_data.loc[pf_value_df.index[0] : pf_value_df.index[-1]]
        cagr_nifty = (
            ((nifty_date_range.iloc[-1] / nifty_date_range.iloc[0]) ** (1 / years)) - 1
        ) * 100
        alpha = cagr_strategy - cagr_nifty

        strat_volatility = aligned_strat.std() * np.sqrt(252)
        nifty_volatility = aligned_nifty.std() * np.sqrt(252)

        sharpe_ratio = (
            (cagr_strategy / 100 - RISK_FREE_RATE_ANNUAL) / strat_volatility
            if strat_volatility > 0
            else 0
        )

        downside_returns = aligned_strat[aligned_strat < 0]
        downside_volatility = (
            downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        )
        sortino_ratio = (
            (cagr_strategy / 100 - RISK_FREE_RATE_ANNUAL) / downside_volatility
            if downside_volatility > 0
            else float("inf")
        )

        covariance = aligned_strat.cov(aligned_nifty) if len(aligned_strat) > 1 else 0
        beta = covariance / aligned_nifty.var() if aligned_nifty.var() > 0 else 0

        pf_value_df["Peak"] = pf_value_df["Value"].cummax()
        max_drawdown = (
            (pf_value_df["Value"] - pf_value_df["Peak"]) / pf_value_df["Peak"]
        ).min()
        calmar_ratio = (
            cagr_strategy / abs(max_drawdown * 100)
            if max_drawdown < 0
            else float("inf")
        )

        m2_ratio = (sharpe_ratio * nifty_volatility * 100) + (
            RISK_FREE_RATE_ANNUAL * 100
        )

        r_squared = (
            aligned_strat.corr(aligned_nifty) ** 2
            if not aligned_strat.isnull().all() and not aligned_nifty.isnull().all()
            else 0
        )

        gross_gains = aligned_strat[aligned_strat > 0].sum()
        gross_losses = abs(aligned_strat[aligned_strat < 0].sum())
        profit_factor = gross_gains / gross_losses if gross_losses > 0 else float("inf")

        avg_averages = (
            np.mean([t["AveragingEvents"] for t in closed_trades_log])
            if closed_trades_log
            else 0
        )
        max_averages = (
            np.max([t["AveragingEvents"] for t in closed_trades_log])
            if closed_trades_log
            else 0
        )

        print("\n--- Detailed Strategy Metrics ---")
        print(
            f"Backtest Period: {pf_value_df.index[0].date()} to {pf_value_df.index[-1].date()}"
        )
        print(
            f"Final Portfolio Value: ₹{final_portfolio_value:,.2f} (Total Return: {((final_portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%)"
        )
        total_dividend_income = 0
        if not transactions_log_df.empty:
            dividend_log = transactions_log_df[
                transactions_log_df["Type"] == "DIVIDEND"
            ].copy()
            if not dividend_log.empty:
                dividend_log["TotalDividend"] = (
                    dividend_log["Price"] * dividend_log["Quantity"]
                )
                total_dividend_income = dividend_log["TotalDividend"].sum()
                # Save the detailed dividend log
                dividend_log_filename = f"dividend_log_{sizing_method}.csv"
                dividend_log.to_csv(dividend_log_filename, index=False)
                print(f"Dividend log saved to {dividend_log_filename}")

        print(f"Total Dividend Income: ₹{total_dividend_income:,.2f}")
        print(
            f"Max Capital Deployed: ₹{pf_value_df['Deployed Capital'].max():,.2f} ({ (pf_value_df['Deployed Capital'].max() / pf_value_df['Value'].max()) * 100:.2f}% of peak portfolio value)"
        )
        print("-" * 35)
        print(f"Strategy CAGR: {cagr_strategy:.2f}%")
        if strategy_xirr is not None:
            print(f"Strategy XIRR: {strategy_xirr*100:.2f}%")
        print(f"NIFTYBEES CAGR:  {cagr_nifty:.2f}%")
        print(f"Alpha (Excess Return): {alpha:.2f}%")
        print(f"Beta (vs NIFTYBEES): {beta:.2f}")
        print(f"R-squared: {r_squared:.2%}")
        print(f"Annual Volatility: {strat_volatility*100:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print("-" * 35)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"M2 Ratio: {m2_ratio:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("-" * 35)
        print(f"Avg. Averages per Trade: {avg_averages:.2f}")
        print(f"Max Averages for a Single Trade: {int(max_averages)}")

    if not transactions_log_df.empty:
        # Save the detailed transaction log
        log_filename = "trade_log.csv"
        transactions_log_df.to_csv(log_filename, index=False)
        print(f"\nTrade log saved to {log_filename}")

        transactions_log_df.set_index("Date", inplace=True)
        buys = (
            transactions_log_df[transactions_log_df["Type"].isin(["BUY", "AVG"])]
            .resample("QE")
            .size()
            .rename("Buys")
        )
        sells = (
            transactions_log_df[transactions_log_df["Type"] == "SELL"]
            .resample("QE")
            .size()
            .rename("Sells")
        )
        quarterly_trades = pd.concat([buys, sells], axis=1).fillna(0).astype(int)
        if not quarterly_trades.empty:
            quarterly_trades["Year"] = quarterly_trades.index.year
            quarterly_trades["Quarter"] = quarterly_trades.index.quarter
            trade_pivot = (
                quarterly_trades.pivot_table(
                    index="Year",
                    columns="Quarter",
                    values=["Buys", "Sells"],
                    aggfunc="sum",
                )
                .fillna(0)
                .astype(int)
            )
            print("\n--- Quarterly Trades Grid (Buys / Sells) ---")
            print(trade_pivot.to_string())

    if closed_trades_log:
        trades_df = pd.DataFrame(closed_trades_log)
        if not trades_df.empty:
            top_5_avg_trades = (
                trades_df.sort_values(by="AveragingEvents", ascending=False)
                .head(5)
                .copy()
            )
            top_5_avg_trades["EntryDate"] = top_5_avg_trades["EntryDate"].dt.strftime(
                "%Y-%m-%d"
            )
            top_5_avg_trades["ExitDate"] = top_5_avg_trades["ExitDate"].dt.strftime(
                "%Y-%m-%d"
            )
            print("\n--- Top 5 Trades by Averaging Events ---")
            print(
                top_5_avg_trades[
                    ["Ticker", "EntryDate", "ExitDate", "AveragingEvents"]
                ].to_string(index=False)
            )

        # --- Top 5 Profitable Trades by Percentage ---
        trades_df["PnL_%"] = (
            (trades_df["ExitPrice"] - trades_df["AvgCost"]) / trades_df["AvgCost"]
        ) * 100
        profitable_trades = trades_df[trades_df["PnL_%"] > 0].copy()
        if not profitable_trades.empty:
            top_5_percent_trades = profitable_trades.sort_values(
                by="PnL_%", ascending=False
            ).head(5)
            top_5_percent_trades["EntryDate"] = top_5_percent_trades[
                "EntryDate"
            ].dt.strftime("%Y-%m-%d")
            top_5_percent_trades["ExitDate"] = top_5_percent_trades[
                "ExitDate"
            ].dt.strftime("%Y-%m-%d")
            top_5_percent_trades["PnL_%"] = top_5_percent_trades["PnL_%"].map(
                "{:.2f}%".format
            )
            print("\n--- Top 5 Profitable Trades by Percentage ---")
            print(
                top_5_percent_trades[
                    ["Ticker", "EntryDate", "ExitDate", "PnL_%"]
                ].to_string(index=False)
            )

        # --- Top 5 Profitable Trades by Amount ---
        trades_df["PnL_Amount"] = (
            trades_df["ExitPrice"] - trades_df["AvgCost"]
        ) * trades_df["Quantity"]
        profitable_trades_amount = trades_df[trades_df["PnL_Amount"] > 0].copy()
        if not profitable_trades_amount.empty:
            top_5_amount_trades = profitable_trades_amount.sort_values(
                by="PnL_Amount", ascending=False
            ).head(5)
            top_5_amount_trades["EntryDate"] = top_5_amount_trades[
                "EntryDate"
            ].dt.strftime("%Y-%m-%d")
            top_5_amount_trades["ExitDate"] = top_5_amount_trades[
                "ExitDate"
            ].dt.strftime("%Y-%m-%d")
            top_5_amount_trades["PnL_Amount"] = top_5_amount_trades["PnL_Amount"].map(
                "₹{:,.2f}".format
            )
            print("\n--- Top 5 Profitable Trades by Amount ---")
            print(
                top_5_amount_trades[
                    ["Ticker", "EntryDate", "ExitDate", "PnL_Amount"]
                ].to_string(index=False)
            )

    if not transactions_log_df.empty:
        buy_avg_transactions = transactions_log_df[
            transactions_log_df["Type"].isin(["BUY", "AVG"])
        ]
        if not buy_avg_transactions.empty:
            print("\n--- Last 10 Transactions (Buys & Averages) ---")
            last_10_trans_print = buy_avg_transactions.tail(10).copy()
            last_10_trans_print.index = last_10_trans_print.index.strftime("%Y-%m-%d")
            last_10_trans_print["Price"] = last_10_trans_print["Price"].map(
                "{:,.2f}".format
            )
            print(last_10_trans_print.to_string())

    if closed_trades_log:
        trades_df = pd.DataFrame(closed_trades_log)
        if not trades_df.empty:
            print("\n--- Last 10 Closed Trades ---")
            last_10_closed = trades_df.tail(10).copy()
            last_10_closed["PnL_%"] = (
                (last_10_closed["ExitPrice"] - last_10_closed["AvgCost"])
                / last_10_closed["AvgCost"]
            ) * 100
            last_10_closed["EntryDate"] = last_10_closed["EntryDate"].dt.strftime(
                "%Y-%m-%d"
            )
            last_10_closed["ExitDate"] = last_10_closed["ExitDate"].dt.strftime(
                "%Y-%m-%d"
            )
            display_cols = [
                "Ticker",
                "EntryDate",
                "ExitDate",
                "AvgCost",
                "ExitPrice",
                "PnL_%",
                "AveragingEvents",
            ]
            last_10_closed["AvgCost"] = last_10_closed["AvgCost"].map("₹{:,.2f}".format)
            last_10_closed["ExitPrice"] = last_10_closed["ExitPrice"].map(
                "₹{:,.2f}".format
            )
            last_10_closed["PnL_%"] = last_10_closed["PnL_%"].map("{:.2f}%".format)
            print(last_10_closed[display_cols].to_string(index=False))

    # --- Final Holdings ---
    print("\n--- Final Portfolio Holdings ---")
    final_date = pf_value_df.index[-1]
    if not portfolio["holdings"]:
        print("  None")
    else:
        total_market_value = 0
        for ticker, lots in sorted(portfolio["holdings"].items()):
            total_qty = sum(lot["quantity"] for lot in lots)
            avg_price = sum(lot["price"] * lot["quantity"] for lot in lots) / total_qty
            current_price = (
                stock_data.loc[final_date, ticker]
                if pd.notna(stock_data.loc[final_date, ticker])
                else 0
            )
            market_value = total_qty * current_price
            total_market_value += market_value
            print(
                f"  - {ticker}: Qty={total_qty}, AvgCost=₹{avg_price:,.2f}, MktPrice=₹{current_price:,.2f}, MktValue=₹{market_value:,.2f}"
            )
        print("-" * 35)
        print(f"  Total Holdings Value: ₹{total_market_value:,.2f}")

    return pf_value_df


# --- 3. RUN THE BACKTESTS AND STORE RESULTS ---
backtest_results = {}
if RUN_FIXED_SIZING_TEST:
    backtest_results["FIXED"] = run_backtest(sizing_method="FIXED")
if RUN_DYNAMIC_SIZING_TEST:
    backtest_results["DYNAMIC"] = run_backtest(sizing_method="DYNAMIC")

# --- 4. PLOT COMBINED EQUITY CURVE ---
if backtest_results:
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(15, 8))

    if "FIXED" in backtest_results:
        ax1.plot(
            backtest_results["FIXED"].index,
            backtest_results["FIXED"]["Value"],
            color="cyan",
            linewidth=2,
            label="Strategy (Fixed Sizing)",
        )
    if "DYNAMIC" in backtest_results:
        ax1.plot(
            backtest_results["DYNAMIC"].index,
            backtest_results["DYNAMIC"]["Value"],
            color="yellow",
            linewidth=2,
            label="Strategy (Dynamic Sizing)",
        )

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Strategy Value (INR)", fontsize=12)

    first_result_key = list(backtest_results.keys())[0]
    start_date = backtest_results[first_result_key].index[0]
    nifty_start_value = nifty_data.reindex(
        backtest_results[first_result_key].index, method="ffill"
    ).loc[start_date]

    nifty_performance = (
        nifty_data.reindex(backtest_results[first_result_key].index, method="ffill")
        / nifty_start_value
    ) * backtest_results[first_result_key]["Value"].iloc[0]

    ax1.plot(
        nifty_performance.index,
        nifty_performance,
        color="red",
        linestyle="--",
        linewidth=2,
        label="NIFTYBEES Index (Normalized)",
    )
    ax1.legend(loc="upper left")

    ax1.set_title("Comparative Equity Curve: Fixed vs. Dynamic Sizing", fontsize=16)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    plt.show()
