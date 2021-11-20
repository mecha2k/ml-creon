import pandas as pd
import pprint
import platform
import os, re
from datetime import datetime
from dotenv import load_dotenv

if platform.system() == "Windows":
    from creon import Creon

myIndex = [
    "code",
    "종목명",
    "현재가",
    "시가",
    "고가",
    "저가",
    "거래량",
    "거래대금(원)",
    "총상장주식수(천주)",
    "52주최고가",
    "52주최저가",
    "연중주최저가",
    "연중최저가",
    "PER",
    "EPS",
    "자본금(백만원)",
    "액면가",
    "배당률",
    "배당수익률",
    "부채비율",
    "유보율",
    "ROE",
    "매출액증가율",
    "경상이익증가율",
    "순이익증가율",
    "매출액(백만원)",
    "경상이익(원)",
    "당기순이익(원)",
    "BPS",
    "영업이익증가율",
    "영업이익(원)",
    "매출액영업이익률",
    "매출액경상이익률",
    "이자보상비율",
    "결산년월",
    "EBITDA(백만원)",
    "SPS",
    "CFPS",
    "시가총액",
    "시가총액비중",
    "외인비중",
]

if __name__ == "__main__":
    load_dotenv(verbose=True)
    id_ = os.getenv("creon_id")
    pwd = os.getenv("creon_pwd")
    cert = os.getenv("creon_cert")
    creon_path = "C:/ProgramData/CREON/STARTER/coStarter.exe /prj:cp"

    # creon = Creon()
    # conn = creon.connect(id_=id_, pwd=pwd, pwdcert=cert, c_path=creon_path)
    # if conn is True:
    #     print("connection established to creonPlus...")

    # balance = creon.get_balance()
    # print(balance)

    # codes = creon.get_stockcodes(1)  # kospi=1, kosdaq=2
    # print("kospi stock counts: ", len(codes))

    # ticker = "005930"
    # index = codes.index("A" + ticker)

    # status = creon.get_stockstatus(codes[index])
    # print(status)

    # features = creon.get_stockfeatures(codes[index])
    # pprint.pp(features)

    # data = list()
    # index = list()
    # for code in codes:
    #     code = re.findall(r"\d+", code)
    #     index.append(code[0])
    #     data.append(creon.get_stockfeatures(code[0]))
    # mareye_df = pd.DataFrame(data, index=index)
    # mareye_df.index.name = "code"
    # mareye_df.to_pickle("data/mareye.pkl")

    mareye_df = pd.read_pickle("data/mareye.pkl")
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
    market_df.to_csv("data/market_data_src.csv", encoding="utf-8-sig")
    print(market_df.head(10))

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

    # codes = mareye_df.index.values
    # print(len(codes))
    #
    # mask = [code.endswith("0") for code in codes]
    # df = market_df.loc[mask]

    print(market_df["name_y"].isna().sum())
    df = market_df.copy().reset_index().dropna(axis=0).rename(columns={"name_y": "종목명"})
    df = df[myIndex]
    df = df.loc[df["매출액(백만원)"] > 10]
    vol_quantile = df["거래량"].quantile(q=0.3, interpolation="linear")
    df = df.loc[df["거래량"] > vol_quantile]

    df["PBR"] = df["현재가"] / df["BPS"]
    df["PSR"] = df["현재가"] / df["SPS"]
    df["PCR"] = df["현재가"] / df["CFPS"]
    df["자본총계(백만원)"] = (df["시가총액"] * df["PBR"] * 100).astype(int)
    df = df.loc[df["자본총계(백만원)"] > df["자본금(백만원)"]]

    df["PBR_rank"] = df["PBR"].rank(ascending=True)
    df["PSR_rank"] = df["PSR"].rank(ascending=True)
    df["PCR_rank"] = df["PCR"].rank(ascending=True)
    df["PER_rank"] = df["PER"].rank(ascending=True)
    df["DIV_rank"] = df["배당수익률"].rank(ascending=False)
    df["rank_tot"] = df["PBR_rank"] + df["PSR_rank"] + df["PCR_rank"] + df["PER_rank"]

    df = df[
        [
            "종목명",
            "현재가",
            "거래량",
            "자본금(백만원)",
            "자본총계(백만원)",
            "영업이익증가율",
            "영업이익(원)",
            "매출액영업이익률",
            "EBITDA(백만원)",
            "시가총액",
            "배당수익률",
            "부채비율",
            "이자보상비율",
            "결산년월",
            "PBR",
            "PSR",
            "PCR",
            "PER",
            "ROE",
            "PBR_rank",
            "PSR_rank",
            "PCR_rank",
            "PER_rank",
            "DIV_rank",
            "rank_tot",
        ]
    ].sort_values(by="rank_tot", ascending=True)

    df.to_csv("data/market_data.csv", encoding="utf-8-sig")
    print(df.head(20))
    print(len(df))
