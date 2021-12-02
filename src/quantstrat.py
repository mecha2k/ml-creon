import numpy as np
import pandas as pd
import quantstats as qs
import time

from datetime import datetime
from fnspace.index import fnspaceItems, creonIndex, marcapIndex, fnspaceNames


def find_fscore_stocks(df):
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

    return df.sort_values(by=["fscore_tot"], axis=0, ascending=False)


def find_low_value_stocks(df):
    df = df.loc[df["매출액"] > 10]
    vol_quantile = df["Volume"].quantile(q=0.3, interpolation="linear")
    df = df.loc[df["Volume"] > vol_quantile]
    equ_quantile = df["자본총계"].quantile(q=0.05, interpolation="linear")
    df = df.loc[df["자본총계"] > equ_quantile]
    df = df.loc[df["자본총계"] > df["자본금"]]

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
    )

    return df.sort_values(by=["rank_tot"], axis=0, ascending=True)


class QuantStrat:
    def __init__(self, stock_no, start):
        self.stock_no = stock_no
        self.start = start
        self.fs_df = None
        self.bm_df = None
        self.fdr_df = None
        self.creon_df = None
        self.stocks = None
        self.times = None
        self.annual = None
        self.bm_yields = None
        self.mddmax = None

        self.get_investing_data()

    def get_investing_data(self):
        df = pd.read_pickle(f"fnspace/data/fs_company_all_2021.pkl")
        df = (
            df.reset_index()
            .rename(columns={"level_0": "Code"})
            .drop("level_1", axis=1)
            .set_index("DATE")
            .rename(columns=fnspaceNames)
        )
        df.index = pd.to_datetime(df.index)
        self.fs_df = df

        df = pd.read_pickle("data/market_data.pkl")
        df = df.reset_index().drop_duplicates(subset="code", keep="first")
        df = df.dropna(axis=0).rename(columns={"name_y": "종목명", "code": "Code"})
        self.creon_df = df[creonIndex]

        self.bm_df = pd.read_pickle("data/kospi_index.pkl")
        self.fdr_df = pd.read_pickle("data/large/marcap_data_all_2021.pkl")
        print("investing data loaded...")

    def update_investing_data(self):
        df = marcap_data(start=self.start, end=datetime.now())
        df = df.loc[df["Market"] == "KOSPI"]
        df = df.loc[df["Code"].str.endswith("0")]
        df = df[marcapIndex]
        df.to_pickle("data/large/marcap_data_all_2021.pkl")
        print("marcap data updated...")

        self.get_investing_data()

    def get_investing_yields(self):
        periods, returns = 0, 1
        results = zip(self.times, self.annual, self.mddmax, self.bm_yields)
        for dt, annual, mdd, bm in results:
            periods += 1.0
            returns *= annual
            states = f"annual, cum. yields({dt.year}): {(annual-1)*100:6,.1f}%, "
            states += f"{(returns-1)*100:6,.1f}%,  max MDD: {mdd*100:6,.1f}%, "
            states += f"kospi: {(bm-1)*100:6,.1f}%, alpha: {(annual-bm)*100:5,.1f}%"
            print(states)

        CAGR = (pow(returns, 1 / periods) - 1) * 100
        print(f"\nCAGR : {CAGR:5,.2f}%, mean MDD : {self.stocks['MDD'].mean()*100:5,.1f}%\n")

    def quantstats_reports(self):
        qs.extend_pandas()
        for dtime in pd.date_range(self.start, datetime.now(), freq="12MS"):
            if dtime.year == datetime.now().year:
                break

            sday = dtime.strftime("%Y-%m")
            eday = datetime(dtime.year + 1, dtime.month - 1, dtime.day).strftime("%Y-%m")
            bm_df = self.bm_df.loc[sday:eday]
            bm_df = bm_df["close"].pct_change()

            codes = self.stocks.loc[str(dtime.year)]["Code"].values
            for code in codes[:1]:
                stock = self.fdr_df.loc[self.fdr_df["Code"] == code]
                stock = stock.loc[sday:eday]
                title = f"{stock['Name'][0]}({code})"
                stock = stock["Close"].pct_change()
                # qs.reports.metrics(stock, mode="basic")
                # stock.plot_earnings(
                #     savefig=f"data/quantstats/qs_{dtime.year}_{title}.png", start_balance=10000
                # )
                qs.reports.html(
                    returns=stock,
                    benchmark=bm_df,
                    title=title,
                    output=f"data/quantstats/qs_{dtime.year}_{title}.html",
                )

    def prepare_annual_dataframe(self, dtime):
        fs = self.fs_df.loc[str(dtime.year)]
        fs = fs.drop_duplicates(subset="Code", keep="first")

        sday = dtime.strftime("%Y-%m-%d")
        eday = datetime(dtime.year + 1, dtime.month, dtime.day).strftime("%Y-%m-%d")
        df = self.fdr_df.loc[sday:eday]
        bm_df = self.bm_df.loc[sday:eday].copy()
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

        # correction for the values of the last year with current stock prices
        fiscal_time = datetime(dtime.year - 1, 12, 31)
        dft = self.fdr_df.loc[fiscal_time.strftime("%Y-%m")]
        fiscal_day = dft.sort_index().index.unique()[-1].strftime("%Y-%m-%d")
        fiscal_df = dft.loc[fiscal_day][["Code", "Close"]]
        fiscal_df["Date_x"] = fiscal_day

        dft = self.fdr_df.loc[dtime.strftime("%Y-%m")]
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

        return df, bm_rets

    def get_stocks_from_strategy(self):
        stocks, times, annual, bm_yields, mddmax = list(), list(), list(), list(), list()
        for dtime in pd.date_range(self.start, datetime.now(), freq="12MS"):
            if dtime.year == datetime.now().year:
                break

            df, bm_rets = self.prepare_annual_dataframe(dtime=dtime)

            # df = find_low_value_stocks(df)
            df = find_fscore_stocks(df)

            df = df.iloc[: self.stock_no]

            stocks.append(df)
            times.append(dtime)
            mddmax.append(df["MDD"].min())
            bm_yields.append(bm_rets)

            cagr = 0
            for code in df.index.values:
                cagr += (df.at[code, "Yield"]) / stock_no
            annual.append(cagr)

        self.times = times
        self.annual = annual
        self.mddmax = mddmax
        self.bm_yields = bm_yields

        stocks = pd.concat(stocks, keys=times)
        stocks = stocks.reset_index().rename(columns={"level_0": "date"}).set_index("date")
        stocks = stocks.drop("level_1", axis=1)
        stocks.to_pickle("data/analysis_results.pkl")
        self.stocks = stocks


if __name__ == "__main__":
    stock_no = 10
    start = datetime(2012, 5, 1)
    qstrat = QuantStrat(stock_no=stock_no, start=start)
    print(f"start : {start}, stock_no : {stock_no}")

    stime = time.time()
    # qstrat.update_investing_data()
    qstrat.get_stocks_from_strategy()
    qstrat.get_investing_yields()
    # qstrat.stocks = pd.read_pickle("data/analysis_results.pkl")
    qstrat.quantstats_reports()
    print(f"\nexecution time elapsed (sec) : {time.time()-stime}")
