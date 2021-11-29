import pandas as pd
import OpenDartReader
import FinanceDataReader as fdr
import os

from marcap import marcap_data
from datetime import datetime
from dotenv import load_dotenv
from pprint import pprint
from fnspace.index import fnspaceItems, marcapIndex

fnspaceNames = {
    "Free Cash Flow2": "FCF",
    "EBITDA2": "EBITDA",
    "ROE(지배)": "ROE",
    "EPS(지배, Adj.)": "EPS",
    "BPS(지배, Adj.)": "BPS",
    "CPS(Adj.)": "CPS",
    "SPS(Adj.)": "SPS",
    "수정DPS(보통주, 연간현금)": "DPS",
    "현금배당성향(%)": "현금배당성향",
    "현금배당수익률(보통주, 연간, 수정주가)": "현금배당수익률",
    "P/E(Adj., FY End)": "PER",
    "P/B(Adj., FY End)": "PBR",
    "P/C(Adj., FY End)": "PCR",
    "P/S(Adj., FY End)": "PSR",
    "P/FCF2(Adj., FY End)": "P/FCF",
    "PEG(Adj., FY End, YoY)": "PEG",
    "EV/EBITDA2": "EV/EBITDA",
}


def get_stock_price_fdr(codes, start):
    for dt in pd.date_range(start=start, end=datetime.now(), freq="12MS"):
        fdr_df = list()
        for code in codes:
            df = fdr.DataReader(code, datetime(dt.year, 1, 1), datetime(dt.year, 12, 31))
            if df.empty is False:
                fdr_df.append(df)
                print(f"code: {code} is appended...")
        fdr_df = pd.concat(fdr_df, keys=codes, names=["code"])
        fdr_df.to_pickle(f"data/fdr_stock_{dt.year}.pkl")
        print(f"fdr_stock_{dt.year} file (stocks: {len(fdr_df)}) saved...")


def get_stock_price_fdr_file(start):
    fdr_df = list()
    for dt in pd.date_range(start=start, end=datetime.now(), freq="12MS"):
        df = pd.read_pickle(f"data/fdr_stock_{dt.year}.pkl")
        fdr_df.append(df)
    return pd.concat(fdr_df)


def get_investing_info_data():
    fs_df = pd.read_pickle(f"fnspace/data/fs_company_all_2021.pkl")
    fs_df = (
        fs_df.reset_index()
        .rename(columns={"level_0": "code"})
        .drop("level_1", axis=1)
        .set_index("DATE")
        .rename(columns=fnspaceNames)
    )
    fs_df.index = pd.to_datetime(fs_df.index)

    market_df = pd.read_pickle("data/market_data.pkl")
    market_df = market_df.reset_index().drop_duplicates(subset="code", keep="first")
    market_df = market_df.dropna(axis=0).rename(columns={"name_y": "종목명"})
    market_df = market_df[marcapIndex]

    fdr_df = get_stock_price_fdr_file(start=datetime(2000, 1, 1))
    fdr_df = fdr_df.reset_index().set_index("Date")

    # codes = market_df["code"].values
    # names = market_df["종목명"].values
    #
    # get stock prices from financeDataReader yearly
    # get_stock_price_fdr(codes, start)

    # df = marcap_data(start="2010-01", end="2010-03")
    # df = df.loc[df["Market"] == "KOSPI"]
    # df = df.loc[df["Code"].str.endswith("0")]
    # print(df)

    # stocks = fdr.StockListing("KOSPI")
    # stocks.to_pickle("data/stock_kospi.pkl")
    # stocks = pd.read_pickle("data/stock_kospi.pkl")
    # stocks = stocks.loc[stocks["Market"] == "KOSPI"].dropna(axis=0)
    # stocks = stocks.loc[stocks["Symbol"].str.endswith("0")]
    # print(stocks)

    return fs_df, market_df, fdr_df


def analyze_strategy(stock_no, fdr_df, fs_df, start):
    stocks, times, annual_yield = list(), list(), list()
    for dtime in pd.date_range(start, datetime.now(), freq="12MS"):
        if dtime.year == datetime.now().year:
            break

        fs = fs_df.loc[str(dtime.year)]
        fs = fs.drop_duplicates(subset="code", keep="first")
        fs = fs.loc[fs["매출액"] > 10]

        sday = dtime.strftime("%Y-%m-%d")
        eday = datetime(dtime.year + 1, dtime.month, dtime.day).strftime("%Y-%m-%d")
        df = fdr_df.loc[sday:eday]

        # CAGR (Compound Annual Growth Rate)
        dfs = list()
        codes = df["code"].unique()
        for code in codes:
            dfc = df.loc[df["code"] == code].resample("MS").first()
            dfc["Yield"] = (1 + dfc["Close"].pct_change()).cumprod()
            dfs.append(dfc.iloc[-1])
        dfs = pd.concat(dfs, keys=codes)
        dfs = dfs.unstack(level=-1)

        df = pd.merge(fs, dfs, how="inner", on="code")
        df = pd.merge(df, market_df[["code", "종목명"]], how="inner", on="code")
        df = df.set_index("code")

        vol_quantile = df["Volume"].quantile(q=0.3, interpolation="linear")
        df = df.loc[df["Volume"] > vol_quantile]
        equ_quantile = df["자본총계"].quantile(q=0.05, interpolation="linear")
        df = df.loc[df["자본총계"] > equ_quantile]
        df = df.loc[df["자본총계"] > df["자본금"]]

        df = df.loc[df["PBR"] > 0.5]
        df = df.loc[df["PCR"] > 2.0]
        df = df.loc[df["PER"] > 5.0]
        df = df.loc[df["PEG"] > 0.0]

        df["PBR_rank"] = df["PBR"].rank(ascending=True)
        df["PSR_rank"] = df["PSR"].rank(ascending=True)
        df["PCR_rank"] = df["PCR"].rank(ascending=True)
        df["PER_rank"] = df["PER"].rank(ascending=True)
        df["PEG_rank"] = df["PEG"].rank(ascending=True)
        df["DIV_rank"] = df["현금배당수익률"].rank(ascending=False)
        df["EV_rank"] = df["EV"].rank(ascending=False)
        df["rank_tot"] = (
            df["PBR_rank"]
            + df["PSR_rank"]
            + df["PCR_rank"]
            + df["PER_rank"]
            + df["PEG_rank"]
            + df["DIV_rank"]
            # + df["EV_rank"]
        )

        df = df.sort_values(by=["rank_tot"], axis=0, ascending=True)
        df = df.iloc[:stock_no]
        stocks.append(df)
        times.append(dtime)

        cagr = 0
        for code in df.index.values:
            cagr += (df.at[code, "Yield"]) / stock_no
        annual_yield.append(cagr)

    stocks = pd.concat(stocks, keys=times)
    stocks = stocks.reset_index().rename(columns={"level_0": "date"}).set_index("date")

    return {"stocks": stocks, "yield": zip(times, annual_yield)}


def investing_yields(results):
    df = results["stocks"]
    df.to_csv("data/analysis_results.csv", encoding="utf-8-sig")

    returns = 1
    periods = 0
    for dt, annual in results["yield"]:
        periods += 1.0
        returns *= annual
        print(
            f"annual, cum. yields ({dt.year}) : {(annual-1)*100:8,.1f}%, {(returns-1)*100:8,.1f}%"
        )

    CAGR = (pow(returns, 1 / periods) - 1) * 100
    print(f"\nCAGR : {CAGR:5,.2f}%\n")


def confirm_strategy(fdr_df, fs_df):
    df = pd.read_csv("data/analysis_results.csv")
    print(df)

    # codes = ["084690", "001120"]
    # df = get_stock_price_fdr_file(start=start)
    # df = df.reset_index().set_index("Date")
    # df = df.sort_index().loc["2020-5-1":"2021-5-30"]
    # for code in codes:
    #     dfc = df.loc[df["code"] == code].resample("MS").first()
    #     dfc["Yield"] = (1 + dfc["Close"].pct_change()).cumprod()
    #     print(dfc)

    # df = fdr.DataReader("003690", datetime(2020, 4, 1), datetime(2021, 4, 1))
    # df = df.drop("Change", axis=1).resample("MS").first()
    # df["Change"] = (1 + df["Close"].pct_change()).cumprod()
    # print(df)


if __name__ == "__main__":
    stock_no = 10
    start = datetime(2012, 5, 1)

    fs_df, market_df, fdr_df = get_investing_info_data()
    # for m in range(3, 4):
    #     start = datetime(2012, m, 1)
    #     print(f"start : {start}")
    #     results = analyze_strategy(stock_no, fdr_df, fs_df, start=start)
    #     investing_yields(results)

    confirm_strategy(fdr_df, fs_df)
