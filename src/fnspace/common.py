import pandas as pd
import FinanceDataReader as fdr
import os

from datetime import datetime
from dotenv import load_dotenv

import index


def get_fs_accounts_info(api_key):
    aid = "A000002"  # 재무정보
    url = f"https://www.fnspace.com/Api/ItemListApi?key={api_key}&format=json&apigb=" + aid
    data = pd.read_json(url, typ="series")
    return pd.DataFrame(data["dataset"])


def get_queries(api_url, query):
    query = query[: len(query) - 1]
    data = pd.read_json(api_url + query, typ="series")
    if data["success"] == "true":
        df = data["dataset"][0]
        df = (
            pd.DataFrame(df["DATA"])
            .drop(["YYMM", "FS_YEAR", "FS_MONTH", "FS_QTR", "MAIN_TERM"], axis=1)
            .set_index("DATE")
        )
    else:
        df = None
        print(data["errmsg"])
    return df


def get_company_fs(code, start, end, items, api_key):
    api_url = f"https://www.fnspace.com/Api/FinanceApi?key={api_key}&format=json&code="
    api_url += f"{code}&consolgb=M&annualgb=A&fraccyear={start}&toaccyear={end}&item="

    query = ""
    fs_df = list()
    for item in items:
        query += f"{item},"
        index = items.index(item)
        if index % 20 == 0:
            df = get_queries(api_url, query)
            if df is not None:
                fs_df.append(df)
            else:
                return None
            query = ""

    query = ""
    for item in items[(len(items) // 20) * 20 + 1 :]:
        query += f"{item},"
    df = get_queries(api_url, query)
    if df is not None:
        fs_df.append(df)

    return pd.concat(fs_df, axis=1)


if __name__ == "__main__":
    load_dotenv(verbose=True)
    api_key = os.getenv("fnspace_key")

    start = datetime(2012, 1, 1)
    end = datetime(2020, 1, 1)

    items_df = get_fs_accounts_info(api_key)
    items_df = items_df.loc[items_df["ITEM_CD"].isin(index.fnspaceItems.keys())]
    items = list(items_df["ITEM_CD"].values)
    names = list(items_df["ITEM_NM_KOR"].values)
    items_dict = dict(zip(items, names))

    market_df = pd.read_pickle("../data/market_data.pkl")
    market_df = market_df.reset_index().drop_duplicates(subset="code", keep="first")
    market_df = market_df.dropna(axis=0).rename(columns={"name_y": "종목명"})
    market_df = market_df[index.marcapIndex]
    print(market_df)

    fs_df = list()
    timenow = datetime.now()
    codes = market_df["code"].values
    # for code in codes:
    #     df = get_company_fs(
    #         code="A" + code, start=start.year, end=end.year, items=items, api_key=api_key
    #     )
    #     if df is not None:
    #         df = df.rename(columns=items_dict)
    #         df.to_pickle(f"data/fs_{code}_{timenow.year}.pkl")
    #         fs_df.append(df)
    #         print(f"fs_{code} is appended...")

    for code in codes:
        df = pd.read_pickle(f"data/fs_{code}_{timenow.year}.pkl")
        df = df[list(index.fnspaceItems.values())].reset_index()
        fs_df.append(df)
    fs_df = pd.concat(fs_df, keys=codes)
    fs_df.to_pickle(f"data/fs_company_all_{timenow.year}.pkl")
    print(fs_df.info())
    print(fs_df.head())
