import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import stratcollect
import scipy.optimize as sco

from datetime import datetime
from icecream import ic
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
from fnspace.index import fnspaceItems, creonIndex, marcapIndex, fnspaceNames

plt.style.use("ggplot")
plt.rcParams["font.family"] = "D2Coding ligature"
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.linestyle"] = "-"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["lines.markersize"] = 5
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.set_cmap("cubehelix")
sns.set_palette("cubehelix")
# warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)
COLORS = [plt.cm.cubehelix(x) for x in [0.1, 0.3, 0.5, 0.7]]


class QuantStrat:
    def __init__(self, stock_no, start):
        self.stock_no = stock_no
        self.start = start
        self.fs_df = None
        self.bm_df = None
        self.fdr_df = None
        self.creon_df = None
        self.stocks = None
        self.assets = None
        self.weights = None
        self.benchmark = None
        self.times = None
        self.fstock_list = list()

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
        periods, cumulative, mdd_avg = 0, 1, 0
        for df, wg, dt, bm in zip(self.assets, self.weights, self.times, self.benchmark):
            cagr = 0
            for code, year_rets in zip(df["Code"].values, df["Yield"].values):
                cagr += year_rets * wg[code]
            mdd_max = df["MDD"].min()
            mdd_avg = df["MDD"].mean()

            periods += 1.0
            cumulative *= cagr
            states = f"연간,누적수익률({dt.year}년,{len(df)}종목): {(cagr-1)*100:6,.1f}%, "
            states += f"{(cumulative-1)*100:6,.1f}%, 최대MDD: {mdd_max*100:6,.1f}%, "
            states += f"코스피: {(bm-1)*100:6,.1f}%, 알파: {(cagr-bm)*100:5,.1f}%"
            print(states)

        CAGR = (pow(cumulative, 1 / periods) - 1) * 100
        print(f"\nCAGR : {CAGR:5,.2f}%, mean MDD : {mdd_avg*100:5,.1f}%\n")

    def quantstats_reports(self, nstock=1):
        qs.extend_pandas()
        for dtime in pd.date_range(self.start, datetime.now(), freq="12MS"):
            if dtime.year == datetime.now().year:
                break

            sday = dtime.strftime("%Y-%m")
            eday = datetime(dtime.year + 1, dtime.month - 1, dtime.day).strftime("%Y-%m")
            bm_df = self.bm_df.loc[sday:eday]
            bm_df = bm_df["close"].pct_change()

            codes = self.stocks.loc[str(dtime.year)]["Code"].values
            for code in codes[:nstock]:
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

    def plot_stock_annual_returns(self):
        if self.stocks is None or self.stocks.empty:
            self.stocks = pd.read_pickle("data/analysis_results.pkl")

        for dtime in pd.date_range(self.start, datetime.now(), freq="12MS"):
            if dtime.year == datetime.now().year:
                break

            sday = dtime.strftime("%Y-%m")
            eday = datetime(dtime.year + 1, dtime.month - 1, dtime.day).strftime("%Y-%m")
            bm_df = self.bm_df.loc[sday:eday]
            bm_df = ((1 + bm_df["close"].pct_change()).cumprod() - 1) * 100

            stock_dict = dict()
            stock_dict["KOSPI"] = bm_df
            codes = self.stocks.loc[str(dtime.year)]["Code"].values
            for code in codes:
                stock = self.fdr_df.loc[self.fdr_df["Code"] == code]
                stock = stock.loc[sday:eday]
                title = f"{stock['Name'][0]}({code})"
                stock = ((1 + stock["Close"].pct_change().dropna(axis=0)).cumprod() - 1) * 100
                stock_dict[title] = stock

            stock_df = pd.concat(stock_dict.values(), keys=stock_dict.keys())
            stock_df = stock_df.unstack(level=0).dropna(axis=0)

            fig, ax = plt.subplots()
            ax = stock_df.plot(title=f"{dtime.year}년 종목별 수익률")
            ax.plot(bm_df, color="blue", linewidth="5")
            ax.set(xlabel="Date", ylabel="Returns (%)")
            ax.xaxis.set_major_locator(MonthLocator())
            date_form = DateFormatter("%y-%m")
            ax.xaxis.set_major_formatter(date_form)
            plt.xticks(rotation=45)
            plt.grid(alpha=0.5, linestyle="-")
            plt.savefig(f"images/qs_{dtime.year}_stocks.png", bbox_inches="tight")
            plt.close(fig)

    def find_momentum_stocks(self, dtime):
        sday = datetime(dtime.year - 1, dtime.month, dtime.day).strftime("%Y-%m")
        eday = datetime(dtime.year, dtime.month - 1, dtime.day).strftime("%Y-%m")
        df = self.fdr_df[sday:eday].reset_index().set_index("Code")
        codes = df.index.drop_duplicates().values
        stocks = list()
        for code in codes:
            dfs = df.loc[[code]].sort_values(by="Date")[["Date", "Close"]].set_index("Date")
            dfs["1y_rets"] = (1 + dfs["Close"].pct_change()).cumprod()
            stocks.append(dfs.iloc[-1])
        df = pd.concat(stocks, keys=codes, names=["Code", "Items"])
        df = df.unstack(level=-1)
        df = df.sort_values(by="1y_rets", ascending=False)

        return df

    def get_stocks_from_strategy(self, rankfunc):
        stocks, assets, weights, times, benchmark = [], [], [], [], []
        for dtime in pd.date_range(self.start, datetime.now(), freq="12MS"):
            if dtime.year == datetime.now().year:
                break

            mom_df = self.find_momentum_stocks(dtime=dtime).reset_index()
            df, bm_rets = self.prepare_annual_dataframe(dtime=dtime)
            df = pd.merge(df, mom_df, how="inner", on="Code")
            df = rankfunc(df)
            asset_df = df.iloc[: self.stock_no]

            stocks.append(df)
            times.append(dtime)
            assets.append(asset_df)
            benchmark.append(bm_rets)
            codes = asset_df["Code"].values
            weight = self.stock_no * [1 / self.stock_no]
            weight = {key: value for key, value in zip(codes, weight)}
            weights.append(weight)

        stocks = pd.concat(stocks, keys=times)
        stocks = stocks.reset_index().rename(columns={"level_0": "date"}).set_index("date")
        stocks = stocks.drop("level_1", axis=1)
        stocks.to_pickle("data/analysis_results.pkl")

        self.times = times
        self.stocks = stocks
        self.assets = assets
        self.weights = weights
        self.benchmark = benchmark

    def get_asset_allocation(self, dtime, stocks, plot=False):
        sday = datetime(dtime.year - 1, dtime.month, dtime.day).strftime("%Y-%m")
        eday = datetime(dtime.year, dtime.month - 1, dtime.day).strftime("%Y-%m")
        df = self.fdr_df[sday:eday].reset_index().set_index(["Code", "Date"])
        codes = stocks["Code"].values
        names = stocks["Name"].values
        df = df.loc[codes][["Close"]].unstack(level=0).droplevel(level=0, axis=1)
        df = df.pct_change().dropna()

        npfs = 10 ** 5
        ndays = len(df.index)
        ncodes = len(codes)
        avg_df = df.mean(axis=0) * ndays
        cov_df = df.cov(ddof=1) * ndays

        # Calculate portfolio metrics:
        weights = np.random.random(size=(npfs, ncodes))
        weights_sum = np.sum(weights, axis=1).reshape(-1, 1)
        weights /= weights_sum
        pf_ret = np.dot(weights, avg_df)

        pf_vol = list()
        for i in range(len(weights)):
            pf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_df, weights[i]))))
        pf_vol = np.array(pf_vol)
        pf_sharpe = pf_ret / pf_vol
        pf_df = pd.DataFrame({"returns": pf_ret, "volatility": pf_vol, "sharpe_ratio": pf_sharpe})

        # Locate the points creating the Efficient Frontier:
        npoints = 100
        pf_vol_ef = []
        inds_to_skip = []
        pf_ret_ef = np.linspace(pf_df.returns.min(), pf_df.returns.max(), npoints)
        pf_ret_ef = np.round(pf_ret_ef, 2)
        pf_ret = np.round(pf_ret, 2)
        for point_index in range(npoints):
            if pf_ret_ef[point_index] not in pf_ret:
                inds_to_skip.append(point_index)
                continue
            matched_ind = np.where(pf_ret == pf_ret_ef[point_index])
            pf_vol_ef.append(np.min(pf_vol[matched_ind]))
        pf_ret_ef = np.delete(pf_ret_ef, inds_to_skip)

        max_sharpe_ind = np.argmax(pf_df.sharpe_ratio)
        max_sharpe_pf = pf_df.loc[max_sharpe_ind]

        min_vol_ind = np.argmin(pf_df.volatility)
        min_vol_pf = pf_df.loc[min_vol_ind]

        wgts = weights[np.argmax(pf_df.sharpe_ratio)]
        df = pd.DataFrame({"name": names, "code": codes, "weight": wgts}).sort_values(
            by="weight", ascending=False
        )

        if plot is False:
            return df

        print("Maximum Sharpe Ratio portfolio ----")
        print("Performance")
        for index, value in max_sharpe_pf.items():
            print(f"{index}: {100 * value:.2f}% ", end="", flush=True)
        print("\n")

        fig, ax = plt.subplots()
        pf_df.plot(
            kind="scatter",
            x="volatility",
            y="returns",
            c="sharpe_ratio",
            cmap="RdYlGn",
            edgecolors="black",
            ax=ax,
        )
        ax.plot(pf_vol_ef, pf_ret_ef, color="blue", linestyle="dashed", linewidth=2)
        ax.scatter(
            x=max_sharpe_pf.volatility,
            y=max_sharpe_pf.returns,
            c="black",
            marker="*",
            s=200,
            label="Max Sharpe Ratio",
        )
        ax.scatter(
            x=min_vol_pf.volatility,
            y=min_vol_pf.returns,
            c="black",
            marker="P",
            s=200,
            label="Minimum Volatility",
        )
        for ind in range(ncodes):
            annots = f"{names[ind]}({codes[ind]}):{wgts[ind]*100:.1f}%"
            x = np.sqrt(cov_df.iloc[ind, ind])
            y = avg_df[ind]
            color = "orangered" if wgts[ind] > 0.05 else "black"
            ax.scatter(
                x=x,
                y=y,
                marker="o",
                s=50,
                color=color,
                label=annots,
            )
            ax.annotate(
                annots,
                (x, y - 0.03),
                size=6,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=1.0),
            )
        ax.axhline(y=0, color="red", linewidth=0.5, linestyle="dashed")
        ax.legend(loc="best", fontsize=8, shadow=True, edgecolor="black")
        ax.set(xlabel="Volatility", ylabel="Expected Returns", title="Efficient Frontier")
        plt.savefig(f"images/asset_alloc_{dtime.year}.png", bbox_inches="tight")

        return df

    def optimize_stocks_from_MPT(self, weight=0.05):
        assets, weights = list(), list()
        for dtime in pd.date_range(self.start, datetime.now(), freq="12MS"):
            if dtime.year == datetime.now().year:
                break

            stocks = self.stocks.loc[dtime.strftime("%Y-%m")].copy()
            stocks = stocks.sort_values(by="rank_tot", ascending=True)

            df = stocks.iloc[: 2 * self.stock_no]
            asset_df = self.get_asset_allocation(dtime, df, plot=True)
            asset_df = asset_df.loc[asset_df["weight"] > weight]
            codes = asset_df["code"].values
            w_dict = {key: value for key, value in zip(codes, asset_df["weight"].values)}
            assets.append(df.loc[df["Code"].isin(codes)])
            weights.append(w_dict)

        self.assets = assets
        self.weights = weights


if __name__ == "__main__":
    stock_no = 10
    start = datetime(2012, 5, 1)
    qstrat = QuantStrat(stock_no=stock_no, start=start)
    print(f"start : {start}, stock_no : {stock_no}")

    stime = time.time()
    # qstrat.update_investing_data()
    qstrat.get_stocks_from_strategy(stratcollect.find_low_value_stocks)
    # qstrat.optimize_stocks_from_MPT()
    qstrat.get_investing_yields()
    qstrat.plot_stock_annual_returns()
    # qstrat.quantstats_reports()

    print(f"\nexecution time elapsed (sec) : {time.time()-stime}")
