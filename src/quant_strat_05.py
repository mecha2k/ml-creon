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
from fnspace.index import fnspaceItems, creonIndex, marcapIndex, fnspaceNames


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
    creon_df = creon_df[creonIndex]

    fdr_df = pd.read_pickle("data/large/marcap_data_all_2021.pkl")

    # df = marcap_data(start="2010-01-01", end="2021-12-31")
    # df = df.loc[df["Market"] == "KOSPI"]
    # fdr_df = df.loc[df["Code"].str.endswith("0")]
    # fdr_df = fdr_df[marcapIndex]
    # fdr_df.to_pickle("data/large/marcap_data_all_2021.pkl")
    # print(fdr_df.info())

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
            dfc = df.loc[df["Code"] == code].reset_index().rename(columns={"Date": "eDate"})
            ret = dfc["Close"].pct_change().dropna()
            cumulative = (1 + ret).cumprod()
            highmark = cumulative.cummax()
            dfc["Yield"] = (1 + dfc["Close"].pct_change()).cumprod()
            dfc["MDD"] = np.min(cumulative / highmark - 1)
            dfs.append(dfc.iloc[-1])
        dfs = pd.concat(dfs, keys=codes)
        dfs = dfs.unstack(level=-1)

        df = pd.merge(fs, dfs, how="inner", on="Code")
        df = df.set_index("Code")

        vol_quantile = df["Volume"].quantile(q=0.3, interpolation="linear")
        df = df.loc[df["Volume"] > vol_quantile]
        equ_quantile = df["자본총계"].quantile(q=0.05, interpolation="linear")
        df = df.loc[df["자본총계"] > equ_quantile]
        df = df.loc[df["자본총계"] > df["자본금"]]

        # correction for the values of the last year with current stock prices
        fiscal_time = datetime(dtime.year - 1, 12, 31)
        dft = fdr_df.loc[fiscal_time.strftime("%Y-%m")]
        fiscal_day = dft.sort_index().index.unique()[-1].strftime("%Y-%m-%d")
        fiscal_df = dft.loc[fiscal_day][["Code", "Close"]]
        fiscal_df["Date_x"] = fiscal_day

        dft = fdr_df.loc[dtime.strftime("%Y-%m")]
        start_day = dft.sort_index().index.unique()[0].strftime("%Y-%m-%d")
        start_df = dft.loc[start_day][["Code", "Close"]]
        start_df["Date_y"] = start_day

        ratio_df = pd.merge(fiscal_df, start_df, how="inner", on="Code")
        ratio_df["Ratio"] = ratio_df["Close_y"] / ratio_df["Close_x"]

        df = pd.merge(df, ratio_df, how="inner", on="Code")
        adjustedIndex = ["PBR", "PCR", "PER", "PSR", "PEG", "P/FCF", "EV", "EV/EBITDA"]
        for ind in adjustedIndex:
            ad_ind = ind + "c"
            df[ad_ind] = df[ind] * df["Ratio"]

        df = df.loc[df["PBRc"] > 0.5]
        df = df.loc[df["PCRc"] > 2.0]
        df = df.loc[df["PERc"] > 5.0]
        df = df.loc[df["PEGc"] > 0.0]

        df["PBR_rank"] = df["PBRc"].rank(ascending=True)
        df["PSR_rank"] = df["PSRc"].rank(ascending=True)
        df["PCR_rank"] = df["PCRc"].rank(ascending=True)
        df["PER_rank"] = df["PERc"].rank(ascending=True)
        df["PEG_rank"] = df["PEGc"].rank(ascending=True)
        df["DIV_rank"] = df["현금배당수익률"].rank(ascending=False)
        df["EV_rank"] = df["EVc"].rank(ascending=False)
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

    periods, returns = 0, 1
    for dt, annual, mdd in results["yield"]:
        periods += 1.0
        returns *= annual
        states = f"annual, cum. yields({dt.year}): {(annual-1)*100:6,.1f}%, "
        states += f"{(returns-1)*100:6,.1f}%,  max MDD: {mdd*100:6,.1f}%"
        print(states)

    CAGR = (pow(returns, 1 / periods) - 1) * 100
    print(f"\nCAGR : {CAGR:5,.2f}%\n")


def confirm_strategy(start, fdr_df, fs_df):
    cand_df = pd.read_pickle("data/analysis_results.pkl")
    cand_df.to_csv("data/analysis_results.csv", encoding="utf-8-sig")

    # qs.extend_pandas()
    # sday = start.strftime("%Y-%m-%d")
    # eday = datetime(start.year + 1, start.month, start.day).strftime("%Y-%m-%d")
    # stock = fdr_df.loc[sday:eday]
    #
    # codes = ["120110"]
    # for code in codes:
    #     stock = stock.loc[stock["Code"] == code]
    #     title = f"{stock['Name'][0]} ({code})"
    #     stock = stock["Close"].pct_change()
    #     stock.plot_earnings(savefig="data/qs_earnings.png", start_balance=10000)
    #     qs.reports.html(returns=stock, benchmark=None, title=title, output=f"data/quantstats.html")
    #     qs.reports.metrics(stock, mode="basic")

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


if __name__ == "__main__":
    stock_no = 10
    start = datetime(2012, 5, 1)
    print(f"start : {start}, stock_no : {stock_no}")

    stime = time.time()
    fs_df, creon_df, fdr_df = get_investing_info_data()
    results = analyze_strategy(stock_no, fdr_df, fs_df, start=start)
    investing_yields(results)

    confirm_strategy(start, fdr_df, fs_df)
    print(f"\nexecution time elapsed (sec) : {time.time()-stime}")
