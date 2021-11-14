import pandas as pd
import pprint
import os, re
from datetime import datetime
from dotenv import load_dotenv
from creon import Creon

if __name__ == "__main__":
    load_dotenv(verbose=True)
    id_ = os.getenv("creon_id")
    pwd = os.getenv("creon_pwd")
    cert = os.getenv("creon_cert")
    creon_path = "C:/ProgramData/CREON/STARTER/coStarter.exe /prj:cp"

    creon = Creon()
    conn = creon.connect(id_=id_, pwd=pwd, pwdcert=cert, c_path=creon_path)
    if conn is True:
        print("connection established to creonPlus...")

    # balance = creon.get_balance()
    # print(balance)

    codes = creon.get_stockcodes(1)  # kospi=1, kosdaq=2
    print("kospi stock counts: ", len(codes))

    # ticker = "005930"
    # index = codes.index("A" + ticker)

    # status = creon.get_stockstatus(codes[index])
    # print(status)

    # features = creon.get_stockfeatures(codes[index])
    # pprint.pp(features)

    data = list()
    index = list()
    for code in codes:
        code = re.findall(r"\d+", code)
        index.append(code[0])
        data.append(creon.get_stockfeatures(code[0]))
    mareye_df = pd.DataFrame(data, index=index)
    mareye_df.index.name = "code"
    mareye_df.to_pickle("data/mareye.pkl")
    print(mareye_df.head())
    print(mareye_df.info())

    # marcap = creon.get_marketcap(target="2")  # '1': KOSPI200, '2': 거래소전체, '4': 코스닥전체
    # marcap_df = pd.DataFrame(marcap)
    # marcap_df.set_index("code", inplace=True)
    # marcap_df.to_pickle("data/marcap.pkl")

    marcap_df = pd.read_pickle("data/marcap.pkl")
    print(marcap_df.head())
    print(marcap_df.info())

    market_df = pd.merge(mareye_df, marcap_df, how="left", on="code")
    market_df.to_pickle("data/market_data.pkl")
    market_df.to_csv("data/market_data.csv", encoding="utf-8-sig")
    print(market_df.head())

    start = datetime(2021, 10, 1)
    end = datetime(2021, 12, 31)

    # data = creon.get_chart(code=ticker, n=60)
    # print(data[0])

    # data = creon.get_chart(code=ticker, target="A", unit="D", date_from=start.strftime("%Y%m%d"))
    # print(data[0])
    #
    # df = pd.DataFrame(data)
    # df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    # df.set_index("date", inplace=True)
    # print(df.tail())
