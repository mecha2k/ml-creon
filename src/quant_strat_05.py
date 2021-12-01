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


def quantstats_analysis(start, fdr_df, cand_df):
    qs.extend_pandas()
    kospi_df = pd.read_pickle("data/kospi_index.pkl")

    for dtime in pd.date_range(start, datetime.now(), freq="12MS"):
        if dtime.year == datetime.now().year:
            break

        sday = dtime.strftime("%Y-%m")
        eday = datetime(dtime.year + 1, dtime.month - 1, dtime.day).strftime("%Y-%m")
        bm_df = kospi_df.loc[sday:eday]
        bm_df.to_csv("data/bm_df_kospi.csv")
        bm_df = bm_df["close"].pct_change()

        codes = cand_df.loc[str(dtime.year)]["Code"].values
        codes = codes[:1]
        for code in codes:
            stock = fdr_df.loc[fdr_df["Code"] == code]
            stock = stock.loc[sday:eday]
            title = f"{stock['Name'][0]}({code})"
            stock = stock["Close"].pct_change()
            qs.reports.metrics(stock, mode="basic")
            stock.plot_earnings(
                savefig=f"data/quantstats/qs_{dtime.year}_{title}.png", start_balance=10000
            )
            qs.reports.html(
                returns=stock,
                benchmark=bm_df,
                title=title,
                output=f"data/quantstats/qs_{dtime.year}_{title}.html",
            )


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
    kospi_df = pd.read_pickle("data/kospi_index.pkl")

    stocks, times, annual, bm_yields, mddmax = list(), list(), list(), list(), list()
    for dtime in pd.date_range(start, datetime.now(), freq="12MS"):
        if dtime.year == datetime.now().year:
            break

        fs = fs_df.loc[str(dtime.year)]
        fs = fs.drop_duplicates(subset="Code", keep="first")
        fs = fs.loc[fs["매출액"] > 10]

        sday = dtime.strftime("%Y-%m-%d")
        eday = datetime(dtime.year + 1, dtime.month, dtime.day).strftime("%Y-%m-%d")
        df = fdr_df.loc[sday:eday]
        bm_df = kospi_df.loc[sday:eday].copy()
        bm_df["yield"] = (1 + bm_df["close"].pct_change()).cumprod()
        bm_rets = bm_df["yield"].iloc[-1]

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

        # calculate F-score (9 indices)
        df["매출총이익률"] = df["매출총이익"] / df["매출액"]
        df["매출총이익률(YoY)"] = df["매출총이익률"].pct_change()
        df["자산회전율"] = df["매출액"] / df["자산총계"]
        df["자산회전율(YoY)"] = df["자산회전율"].pct_change()
        df["부채비율(YoY)"] = -df["부채비율"].pct_change()
        df["ROA(YoY)"] = df["ROA"].pct_change()
        df["영업현금-이익"] = df["영업현금흐름"] - df["영업이익"]

        fscoreIndex = {
            "fscore1": "당기순이익",  # 당기순이익 > 0
            "fscore2": "영업현금흐름",  # 영업현금흐름 > 0
            "fscore3": "ROA(YoY)",  # ROA(YoY) > 0
            "fscore4": "영업현금-이익",  # 영업현금흐름 > 당기순이익
            "fscore5": "부채비율(YoY)",  # 부채비율(YoY) < 0
            "fscore6": "매출총이익률(YoY)",  # 매출총이익률(YoY) > 0
            "fscore7": "자산회전율(YoY)",  # 자산회전율(매출/자산)(YoY) > 0
        }
        for key, value in fscoreIndex.items():
            df[key] = 0
            df.loc[df[value] > 0, key] = 1
        df["fscore_tot"] = 0
        for ind in fscoreIndex.keys():
            df["fscore_tot"] += df[ind]

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
        )

        df = df.sort_values(by=["rank_tot"], axis=0, ascending=True)
        df = df.iloc[:stock_no]
        stocks.append(df)
        times.append(dtime)
        mddmax.append(df["MDD"].min())
        bm_yields.append(bm_rets)

        cagr = 0
        for code in df.index.values:
            cagr += (df.at[code, "Yield"]) / stock_no
        annual.append(cagr)

    stocks = pd.concat(stocks, keys=times)
    stocks = stocks.reset_index().rename(columns={"level_0": "date"}).set_index("date")

    return {"stocks": stocks, "yield": zip(times, annual, mddmax, bm_yields)}


def investing_yields(results):
    df = results["stocks"]
    df.drop("level_1", axis=1).to_pickle("data/analysis_results.pkl")

    periods, returns = 0, 1
    for dt, annual, mdd, bm in results["yield"]:
        periods += 1.0
        returns *= annual
        states = f"annual, cum. yields({dt.year}): {(annual-1)*100:6,.1f}%, "
        states += f"{(returns-1)*100:6,.1f}%,  max MDD: {mdd*100:6,.1f}%, "
        states += f"kospi: {(bm-1)*100:6,.1f}%, alpha: {(annual-bm)*100:5,.1f}%"
        print(states)

    CAGR = (pow(returns, 1 / periods) - 1) * 100
    print(f"\nCAGR : {CAGR:5,.2f}%, mean MDD : {df['MDD'].mean()*100:5,.1f}%\n")


def confirm_strategy(start, fdr_df, fs_df):
    cand_df = pd.read_pickle("data/analysis_results.pkl")
    cand_df.to_csv("data/analysis_results.csv", encoding="utf-8-sig")

    quantstats_analysis(start, fdr_df, cand_df)


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

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
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
