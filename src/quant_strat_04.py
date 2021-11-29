import numpy as np
import pandas as pd
import OpenDartReader
import FinanceDataReader as fdr
import quantstats as qs
import os, time

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
        fdr_df = pd.concat(fdr_df, keys=codes, names=["Code"])
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
        .rename(columns={"level_0": "Code"})
        .drop("level_1", axis=1)
        .set_index("DATE")
        .rename(columns=fnspaceNames)
    )
    fs_df.index = pd.to_datetime(fs_df.index)

    creon_df = pd.read_pickle("data/market_data.pkl")
    creon_df = creon_df.reset_index().drop_duplicates(subset="code", keep="first")
    creon_df = creon_df.dropna(axis=0).rename(columns={"name_y": "종목명", "code": "Code"})
    creon_df = creon_df[marcapIndex]

    fdr_df = pd.read_pickle("data/large/marcap_data_all_2021.pkl")

    # get stock prices from financeDataReader yearly
    # get_stock_price_fdr(codes, start)
    # financeDataReader stock price data from the year of 2000
    # fdr_df = get_stock_price_fdr_file(start=datetime(2000, 1, 1))
    # fdr_df = pd.read_pickle("data/large/fdr_stock_all_2021.pkl")
    # fdr_df = fdr_df.reset_index().set_index("Date")

    # codes = creon_df["code"].values
    # names = creon_df["종목명"].values

    # df = marcap_data(start="2010-01-01", end="2021-12-31")
    # df = df.loc[df["Market"] == "KOSPI"]
    # df = df.loc[df["Code"].str.endswith("0")]
    # df.to_pickle("data/large/marcap_data_all_2021.pkl")
    # print(df)

    # stocks = fdr.StockListing("KOSPI")
    # stocks.to_pickle("data/stock_kospi.pkl")
    # stocks = pd.read_pickle("data/stock_kospi.pkl")
    # stocks = stocks.loc[stocks["Market"] == "KOSPI"].dropna(axis=0)
    # stocks = stocks.loc[stocks["Symbol"].str.endswith("0")]
    # print(stocks)

    return fs_df, creon_df, fdr_df


def analyze_strategy(stock_no, fdr_df, fs_df, start):
    stocks, times, annual, mddmax = list(), list(), list(), list()
    for dtime in pd.date_range(start, datetime.now(), freq="12MS"):
        if dtime.year == datetime.now().year:
            break

        fs = fs_df.loc[str(dtime.year)]
        fs = fs.drop_duplicates(subset="Code", keep="first")
        fs = fs.loc[fs["매출액"] > 10]

        sday = dtime.strftime("%Y-%m-%d")
        eday = datetime(dtime.year + 1, dtime.month, dtime.day).strftime("%Y-%m-%d")
        df = fdr_df.loc[sday:eday]

        # MDD (Maximum Drawdown)
        # CAGR (Compound Annual Growth Rate)
        dfs = list()
        codes = df["Code"].unique()
        for code in codes:
            dfc = df.loc[df["Code"] == code]
            ret = dfc["Close"].pct_change().dropna()
            cumulative = (1 + ret).cumprod()
            highmark = cumulative.cummax()
            dfc = dfc.resample("MS").first()
            dfc["Yield"] = (1 + dfc["Close"].pct_change()).cumprod()
            dfc["MDD"] = np.min(cumulative / highmark - 1)
            dfs.append(dfc.iloc[-1])
        dfs = pd.concat(dfs, keys=codes)
        dfs = dfs.unstack(level=-1)

        df = pd.merge(fs, dfs, how="inner", on="Code")
        df = pd.merge(df, creon_df[["Code", "종목명"]], how="inner", on="Code")
        df = df.set_index("Code")

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
        mddmax.append(df["MDD"].min())

        cagr = 0
        for code in df.index.values:
            cagr += (df.at[code, "Yield"]) / stock_no
        annual.append(cagr)

    stocks = pd.concat(stocks, keys=times)
    stocks = stocks.reset_index().rename(columns={"level_0": "date"}).set_index("date")

    return {"stocks": stocks, "yield": zip(times, annual, mddmax)}


def investing_yields(results):
    df = results["stocks"]
    df.to_pickle("data/analysis_results.pkl")

    periods, rets = 0, 1
    for dt, an, mdd in results["yield"]:
        periods += 1.0
        rets *= an
        states = f"annual, cum. yields({dt.year}): {(an-1)*100:8,.1f}%, "
        states += f"{(rets-1)*100:8,.1f}%, max MDD: {mdd*100:5,.1f}%"
        print(states)

    CAGR = (pow(rets, 1 / periods) - 1) * 100
    print(f"\nCAGR : {CAGR:5,.2f}%\n")


def confirm_strategy(start, fdr_df, fs_df):
    codes = fs_df["Code"].unique()
    cand_df = pd.read_pickle("data/analysis_results.pkl")
    cand_df.to_csv("data/analysis_results.csv", encoding="utf-8-sig")

    qs.extend_pandas()
    sday = start.strftime("%Y-%m-%d")
    eday = datetime(start.year + 1, start.month, start.day).strftime("%Y-%m-%d")
    stock = fdr_df.loc[sday:eday]

    codes = ["120110"]
    for code in codes:
        stock = stock.loc[fdr_df["Code"] == code]
        stock = stock[["Code", "Name", "Close", "Volume", "Amount"]]
        print(stock)
        stock = stock["Close"].pct_change()
        print(stock)
        stock.plot_earnings(savefig="data/qs_earnings.png", start_balance=10000)
        # qs.reports.html(returns=stock, benchmark=None, output="data/quantstats.html")

        # print(qs.stats.sharpe(stock))
        # print(stock.sharpe())
        # print(stock.monthly_returns())
        # print(stock.max_drawdown())
        #
        # qs.reports.plots(stock, mode="basic")
        # qs.reports.metrics(stock, mode="basic")
        # qs.reports.html(stock, "AAPL", output="quantstats/results/quantstats-aapl.html")

    # KOSPI 지수 가져오기 (벤치마크)
    # 매출액 성장률
    # 부채비율
    # 이자보상비율
    # 거래량 분위
    # 1년간 MDD (투자 전/후)
    # 주가 변동성 (투자 전/후)
    # 종목 상관지수
    # fbprophet 예측 결과
    # 투자 시점 (월)의 차이에 의한 수익률 차이
    # 코스피 기준으로 알파, 샤프 지수 계산

    # F-score (9)
    # 당기순이익이 0 이상 인가?
    # 영업현금흐름이 0 이상 인가?
    # ROA가 전년대비 증가 했는가?
    # 영업현금흐름이 순이익보다 높은가?
    # 부채비율이 전년대비 감소했는가?
    # 유동비율이 전년대비 증가했는가?
    # 당해 신규주식 발행을 하지 않았는가?
    # 매출총이익(매출총이익/매출)이 전년대비 증가했는가?
    # 자산회전율(매출/자산)이 전년대비 증가했는가?

    # codes = ["084690", "001120"]
    # df = get_stock_price_fdr_file(start=start)
    # df = df.reset_index().set_index("Date")
    # df = df.sort_index().loc["2020-5-1":"2021-5-30"]
    # for code in codes:
    #     dfc = df.loc[df["Code"] == code].resample("MS").first()
    #     dfc["Yield"] = (1 + dfc["Close"].pct_change()).cumprod()
    #     print(dfc)

    # stocks = fdr.StockListing('KOSPI')
    df = fdr.DataReader("120110", datetime(2020, 8, 1), datetime(2021, 8, 1))
    df = df.drop("Change", axis=1).resample("MS").first()
    df["Change"] = (1 + df["Close"].pct_change()).cumprod()
    print(df)


if __name__ == "__main__":
    stock_no = 10
    start = datetime(2020, 8, 1)

    stime = time.time()
    fs_df, creon_df, fdr_df = get_investing_info_data()
    for mon in range(7, 8):
        start = datetime(2012, mon + 1, 1)
        print(f"start : {start}")
        results = analyze_strategy(stock_no, fdr_df, fs_df, start=start)
        investing_yields(results)

    # confirm_strategy(start, fdr_df, fs_df)
    print(f"\nexecution time elapsed (sec) : {time.time()-stime}")
