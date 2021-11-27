import pandas as pd
from pandas.io.formats.format import return_docstring
from marcap import marcap_data
from datetime import datetime, timedelta

marcap_ind = {
    "Code": "종목코드",
    "Name": "종목",
    "Close": "종가",
    "Volume": "거래량",
    "Amount": "거래대금",
    "Marcap": "시가총액(백만원)",
    "Stocks": "상장주식수",
    "Market": "시장",
}


def get_marcap_period(start, end):
    data, keys = list(), list()
    for date in pd.date_range(start=start, end=end, freq="AS"):
        date = datetime(date.year, 4, 1)
        df = marcap_data(start=date)
        while len(df) == 0:
            date += timedelta(days=1)
            df = marcap_data(start=date)
        df = df[list(marcap_ind.keys())].reset_index(drop=True)
        data.append(df)
        keys.append(date)
        print(date, len(df))

    return pd.concat(data, keys=keys)


if __name__ == "__main__":
    start = datetime(2000, 1, 1)
    end = datetime(2021, 12, 31)

    # df = get_marcap_period(start=start, end=end)
    # df.to_pickle("data/marcap_period.pkl")

    df = pd.read_pickle("data/marcap_period.pkl")
    print(f"{len(df):,}")

    df = df.loc[df["Code"].str.endswith("0")]
    df = df.loc[df["Market"] == "KOSPI"]
    print(f"{len(df):,}")

    df = df.droplevel(1).rename_axis("Date")
    print(df.info())
    # print(df.groupby("Date").first())
    # for dt in pd.date_range(start=start, end=end, freq="AS"):
    #     data = df.loc[str(dt.year)]
    #     vol_quantile = data["Volume"].quantile(q=0.3, interpolation="linear")
    #     df = df.loc[df["Volume"] > vol_quantile]
    #     print(f"{dt.year} : {len(data):,}")

    # ticker = "005930"
    # df = marcap_data("2020-01-01", "2020-12-31")
    # df = marcap_data(start=start, end=end, code=ticker)
    # df = df.resample(rule="Y").last()
    # df.to_csv("data/marcap.csv", encoding="utf-8-sig")

    # close = int(df.loc["2020", "Close"])
    # marketcap = int(df.loc["2020", "Marcap"])
    # stock_tot = int(marketcap / close)  # 5969782550
    # print(close, marketcap, stock_tot)
