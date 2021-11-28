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


def get_stock_price_fdr(codes, start, end):
    for dt in pd.date_range(start=start, end=end, freq="AS"):
        dt = dt.year
        fdr_df = list()
        for code in codes:
            df = fdr.DataReader(code, datetime(dt, 4, 1), datetime(dt + 1, 4, 1))
            if df.empty is False:
                df = df.drop("Change", axis=1).resample("MS").first()
                # CAGR (Compound Annual Growth Rate)
                df["Change"] = (1 + df["Close"].pct_change()).cumprod()
                fdr_df.append(df.iloc[-1])
        fdr_df = pd.concat(fdr_df, keys=codes, names=["code"])
        fdr_df = fdr_df.unstack(level=-1)
        fdr_df.to_pickle(f"data/fdr_stock_{dt}.pkl")
        print(f"fdr_stock_{dt} file saved...")


def get_stock_price_fdr_file(start, end):
    fdr_df, keys = list(), list()
    for dt in pd.date_range(start=start, end=end, freq="AS"):
        df = pd.read_pickle(f"data/fdr_stock_{dt.year}.pkl")
        fdr_df.append(df)
        keys.append(datetime(dt.year, 4, 1))
    fdr_df = pd.concat(fdr_df, keys=keys, names=["date"])
    fdr_df = fdr_df.reset_index().set_index("date")
    return fdr_df


def analyze_strategy(stock_no, fdr_df, fs_df, start, end):
    stocks, times, annual_yield = list(), list(), list()
    for dtime in pd.date_range(start, end, freq="AS"):
        dt = dtime.year
        fs = fs_df.loc[str(dt)]
        fs = fs.drop_duplicates(subset="code", keep="first")
        fs = fs.loc[fs["매출액"] > 10]
        # codes = fs["code"].values

        df = fdr_df.loc[str(dt)].reset_index().drop("date", axis=1)

        df = pd.merge(fs, df, how="inner", on="code")
        df = pd.merge(df, market_df[["code", "종목명"]], how="inner", on="code")
        df = df.set_index("code")

        vol_quantile = df["Volume"].quantile(q=0.3, interpolation="linear")
        df = df.loc[df["Volume"] > vol_quantile]
        equ_quantile = df["자본총계"].quantile(q=0.05, interpolation="linear")
        df = df.loc[df["자본총계"] > equ_quantile]
        df = df.loc[df["자본총계"] > df["자본금"]]

        df = df.loc[df["PBR"] > 0.5]
        df = df.loc[df["PCR"] > 2.0]
        df = df.loc[df["PER"] > 3.0]
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
            # + df["PER_rank"]
            + df["PEG_rank"]
            # + df["DIV_rank"]
            # + df["EV_rank"]
        )

        df = df.sort_values(by=["rank_tot"], axis=0, ascending=True)
        # df = df[["종목명", "EV_rank", "rank_tot", "Change"]]
        df = df.iloc[:stock_no]
        stocks.append(df)
        times.append(dtime)

        cagr = 0
        for code in df.index.values:
            cagr += (df.at[code, "Change"]) / stock_no
        annual_yield.append(cagr)

    stocks = pd.concat(stocks, keys=times)
    stocks = stocks.reset_index().rename(columns={"level_0": "date"}).set_index("date")

    return {"stocks": stocks, "yield": annual_yield}


if __name__ == "__main__":
    start = datetime(2012, 1, 1)
    end = datetime(2020, 12, 31)

    stock_no = 10

    market_df = pd.read_pickle("data/market_data.pkl")
    market_df = market_df.reset_index().drop_duplicates(subset="code", keep="first")
    market_df = market_df.dropna(axis=0).rename(columns={"name_y": "종목명"})
    market_df = market_df[marcapIndex]
    codes = market_df.index.values

    fs_df = pd.read_pickle(f"fnspace/data/fs_company_all_2021.pkl")
    fs_df = (
        fs_df.reset_index()
        .rename(columns={"level_0": "code"})
        .drop("level_1", axis=1)
        .set_index("DATE")
        .rename(columns=fnspaceNames)
    )
    fs_df.index = pd.to_datetime(fs_df.index)

    # get_stock_price_fdr(codes, start=start, end=end)
    fdr_df = get_stock_price_fdr_file(start=start, end=end)

    results = analyze_strategy(stock_no=stock_no, fdr_df=fdr_df, fs_df=fs_df, start=start, end=end)
    df = results["stocks"][["code", "종목명", "Close", "Change"]]
    df.to_csv("data/analysis_results.csv", encoding="utf-8-sig")
    print(df.tail(10))

    returns = 1
    for annual in results["yield"]:
        returns *= annual
        print(f"annual and total returns : {annual:.2f}, {returns:.2f}")

    CAGR = (pow(returns, 1 / (end.year - start.year + 1)) - 1) * 100
    print(f"\nCAGR : {CAGR:.2f}%")

    # df = fdr.DataReader("003690", datetime(2020, 4, 1), datetime(2021, 4, 1))
    # df = df.drop("Change", axis=1).resample("MS").first()
    # df["Change"] = (1 + df["Close"].pct_change()).cumprod()
    # print(df)
