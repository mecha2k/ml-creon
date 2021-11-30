import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import marketDB


def get_stock_data(stocks_, start="2010-01-04", end="2020-01-27"):
    market = marketDB.MarketDB()
    dataframe_ = pd.DataFrame()
    for stock in stocks_:
        dataframe_[stock] = market.getDailyPrice(stock, start, end)["close"]
    return dataframe_


def eff_frontier(df):
    daily_ret = df.pct_change()
    annual_ret = daily_ret.mean() * 252
    daily_cov = daily_ret.cov()
    annual_cov = daily_cov * 252

    port_ret = []
    port_risk = []
    port_weights = []
    for _ in range(20000):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        returns = np.dot(weights, annual_ret)
        risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        port_ret.append(returns)
        port_risk.append(risk)
        port_weights.append(weights)

    portfolio = {"Returns": port_ret, "Risk": port_risk}
    for i, s in enumerate(stocks):
        portfolio[s] = [weight[i] for weight in port_weights]
    df = pd.DataFrame(portfolio)
    df = df[["Returns", "Risk"] + [s for s in stocks]]

    df.plot.scatter(x="Risk", y="Returns", figsize=(8, 6), grid=True)
    plt.title("Efficient Frontier")
    plt.xlabel("Risk")
    plt.ylabel("Expected Returns")
    plt.show()


def portfolio_optimize(df):
    daily_ret = df.pct_change()
    annual_ret = daily_ret.mean() * 252
    daily_cov = daily_ret.cov()
    annual_cov = daily_cov * 252

    port_ret = []
    port_risk = []
    port_weights = []
    sharpe_ratio = []
    for _ in range(20000):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        returns = np.dot(weights, annual_ret)
        risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        port_ret.append(returns)
        port_risk.append(risk)
        port_weights.append(weights)
        sharpe_ratio.append(returns / risk)

    portfolio = {"Returns": port_ret, "Risk": port_risk, "Sharpe": sharpe_ratio}
    for i, s in enumerate(stocks):
        portfolio[s] = [weight[i] for weight in port_weights]
    df = pd.DataFrame(portfolio)
    df = df[["Returns", "Risk", "Sharpe"] + [s for s in stocks]]

    max_sharpe = df.loc[df["Sharpe"] == df["Sharpe"].max()]
    min_risk = df.loc[df["Risk"] == df["Risk"].min()]

    df.plot.scatter(
        x="Risk",
        y="Returns",
        c="Sharpe",
        cmap="viridis",
        edgecolors="k",
        figsize=(11, 7),
        grid=True,
    )
    plt.scatter(x=max_sharpe["Risk"], y=max_sharpe["Returns"], c="r", marker="*", s=300)
    plt.scatter(x=min_risk["Risk"], y=min_risk["Returns"], c="r", marker="X", s=200)
    plt.title("Portfolio Optimization")
    plt.xlabel("Risk")
    plt.ylabel("Expected Returns")
    plt.show()


def bollinger_band():
    market = marketDB.MarketDB()
    df = market.getDailyPrice("NAVER", "2019-01-02")

    df["MA20"] = df["close"].rolling(window=20).mean()
    df["stddev"] = df["close"].rolling(window=20).std()
    df["upper"] = df["MA20"] + (df["stddev"] * 2)
    df["lower"] = df["MA20"] - (df["stddev"] * 2)
    df = df[19:]

    plt.figure(figsize=(9, 5))
    plt.plot(df.index, df["close"], color="#0000ff", label="Close")
    plt.plot(df.index, df["upper"], "r--", label="Upper band")
    plt.plot(df.index, df["MA20"], "k--", label="Moving average 20")
    plt.plot(df.index, df["lower"], "c--", label="Lower band")
    plt.fill_between(df.index, df["upper"], df["lower"], color="0.9")
    plt.legend(loc="best")
    plt.title("NAVER Bollinger Band (20 day, 2 std)")
    plt.show()


if __name__ == "__main__":
    stocks = ["삼성전자", "SK하이닉스", "현대자동차", "NAVER"]
    dataframe = get_stock_data(stocks, start="2010-01-04", end="2020-01-27")

    eff_frontier(dataframe)
    portfolio_optimize(dataframe)
    bollinger_band()
