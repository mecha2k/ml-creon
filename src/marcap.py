import numpy as np
import pandas as pd
import platform
from datetime import datetime


def marcap_data(start, end=None, code=None):
    start = pd.to_datetime(start)
    end = start if end is None else pd.to_datetime(end)
    df_list = []

    dtypes = {
        "Code": "object",
        "Name": "object",
        "Open": "int64",
        "High": "int64",
        "Low": "int64",
        "Close": "int64",
        "Volume": "int64",
        "Amount": "int64",
        "Changes": "int64",
        "ChangeCode": "object",
        "ChagesRatio": "float",
        "Marcap": "int64",
        "Stocks": "int64",
        "MarketId": "object",
        "Market": "object",
        "Dept": "object",
        "Rank": "int64",
    }

    # Linux: Linux, Mac: Darwin, Windows: Windows
    os_name = platform.system()
    if os_name == "Windows":
        marcap_path = "/Code/python/master/marcap/data/marcap"
    elif os_name == "Darwin":
        marcap_path = "/Users/mecha2k/Documents/Code/python/master/marcap/data/marcap"
    else:
        marcap_path = "/home/mecha2k/code/python/master/marcap/data/marcap"

    for year in range(start.year, end.year + 1):
        try:
            # csv_file = "data/marcap-%s.csv.gz" % (year)
            csv_file = f"{marcap_path}-{year}.csv.gz"
            df = pd.read_csv(csv_file, dtype=dtypes, parse_dates=["Date"])
            df_list.append(df)
        except Exception as e:
            print(e)
            pass
    df_merged = pd.concat(df_list)
    df_merged = df_merged[(start <= df_merged["Date"]) & (df_merged["Date"] <= end)]
    df_merged = df_merged.sort_values(["Date", "Rank"])
    if code:
        df_merged = df_merged[code == df_merged["Code"]]
    df_merged.set_index("Date", inplace=True)
    return df_merged[df_merged["Volume"] > 0]


if __name__ == "__main__":
    # df = marcap_data("2021-1-10", "2021-01-31", code="005930")
    # print(df.tail())

    df = marcap_data("2021-11-10")
    print(df.tail())
