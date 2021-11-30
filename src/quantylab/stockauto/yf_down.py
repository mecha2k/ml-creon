import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime
from icecream import ic


if __name__ == "__main__":
    src_data = "data/yf_data.pkl"
    start = datetime(2020, 1, 1)
    end = datetime(2020, 2, 1)
    tickers = ["AAPL", "MSFT", "GE", "IBM", "AA", "DAL", "UAL", "PEP", "KO"]
    get_data = lambda ticker: yf.download(ticker, start=start, end=end)
    data = map(get_data, tickers)
    print(data)
    data = pd.concat(data, keys=tickers, names=["Ticker", "Date"])
    data.to_pickle(src_data)
    data.info()
    print(data.head())

    # reset the index to make everything columns
    just_closing_prices = data[["Adj Close"]].reset_index()
    ic(just_closing_prices[:5])

    # now pivot Date to the index, Ticker values to columns
    daily_close_px = just_closing_prices.pivot(index="Date", columns="Ticker", values="Adj Close")
    ic(daily_close_px[:5])
