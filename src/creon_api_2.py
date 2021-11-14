import pandas as pd
import pprint
import os
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

    codes = creon.get_stockcodes(1)  # kospi=1, kosdaq=2
    print(codes[:10])
    print("kospi stock counts: ", len(codes))

    ticker = "005930"
    index = codes.index("A" + ticker)

    # status = creon.get_stockstatus(codes[index])
    # print(status)

    features = creon.get_stockfeatures(codes[index])
    pprint.pp(features)

    # marcap = creon.get_marketcap(target="1")  # '1': KOSPI200, '2': 거래소전체, '4': 코스닥전체
    # marcap_df = pd.DataFrame(marcap)
    # marcap_df.set_index("code", inplace=True)
    # marcap_df.to_pickle("data/marcap.pkl")

    marcap_df = pd.read_pickle("data/marcap.pkl")
    print(marcap_df.head())
    print(marcap_df.loc[ticker])

    # balance = creon.get_balance()
    # print(balance)

    # codes = creon.get_chart("005930", n=60)
    # print(codes)
