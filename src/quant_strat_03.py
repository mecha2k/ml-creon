import pandas as pd
import OpenDartReader
import FinanceDataReader as fdr
import os

from marcap import marcap_data
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pprint import pprint
from fnspace.index import fnspaceItems, marcapIndex


if __name__ == "__main__":
    start = datetime(2012, 4, 1)
    end = datetime(2020, 4, 1)

    market_df = pd.read_pickle("data/market_data.pkl")
    market_df = market_df.reset_index().drop_duplicates(subset="code", keep="first")
    market_df = market_df.dropna(axis=0).rename(columns={"name_y": "종목명"})
    market_df = market_df[marcapIndex]

    fs_df = pd.read_pickle(f"fnspace/data/fs_company_all_2021.pkl")
    fs_df = (
        fs_df.reset_index()
        .rename(columns={"level_0": "code"})
        .drop("level_1", axis=1)
        .set_index("code")
    )

    fs_codes = list(fs_df.index.unique())
    ma_codes = list(market_df["code"].values)
    matched = list(set(fs_codes).intersection(set(ma_codes)))
    print(f"No. of codes matched {len(matched)} in market_df {len(fs_codes)}")
    market = market_df.set_index("code")

    fs_codes = fs_codes[:1]
    for code in fs_codes:
        names = market.at[code, "종목명"]
        fnspc = fs_df.loc[code].reset_index(drop=True).set_index("DATE")
        fnspc.index = pd.to_datetime(fnspc.index)
        fnspc = fnspc.loc[str(end.year)]
        creon = market.loc[code]
        print(fnspc.T)
        print(creon)

    # df.to_csv(f"data/fs_{code}_net.csv", encoding="utf-8-sig")
    # print(df)
