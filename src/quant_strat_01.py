import pandas as pd
import OpenDartReader
import FinanceDataReader as fdr
import os

from marcap import marcap_data
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pprint import pprint

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

fs_col_names = {
    "rcept_no": "접수번호",
    "bsns_year": "사업연도",
    "corp_code": "회사코드",
    "stock_code": "종목코드",
    "reprt_code": "보고서코드",
    "account_id": "계정ID",
    "account_nm": "계정명",
    "account_detail": "계정상세",
    "fs_div": "개별/연결구분",
    "fs_nm": "개별/연결명",
    "sj_div": "재무제표구분",
    "sj_nm": "재무제표명",
    "thstrm_nm": "당기명",
    "thstrm_dt": "당기일자",
    "thstrm_amount": "당기금액",
    "thstrm_add_amount": "당기누적금액",
    "frmtrm_nm": "전기명",
    "frmtrm_dt": "전기일자",
    "frmtrm_amount": "전기금액",
    "frmtrm_q_nm": "전기명(분/반기)",
    "frmtrm_q_amount": "전기금액(분/반기)",
    "frmtrm_add_amount": "전기누적금액",
    "bfefrmtrm_nm": "전전기명",
    "bfefrmtrm_dt": "전전기일자",
    "bfefrmtrm_amount": "전전기금액",
    "ord": "계정과목 정렬순서",
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


def get_dart_fs(codes, year):
    print(f"{len(codes)} companies in {year} is downloading...")

    fs_df, fs_all_df = list(), list()
    fs_no, fs_all_no = 0, 0
    for code in codes:
        fs = dart.finstate(code, year)
        fs_all = dart.finstate_all(code, year)
        if fs is not None:
            fs_df.append(fs)
            fs_no += 1
            print(f"fs_net {code} ({fs_no}th) appended...")
        if fs_all is not None:
            fs_all_df.append(fs_all)
            fs_all_no += 1
            print(f"fs_all {code} ({fs_all_no}th) appended...")

    fs_df = pd.concat(fs_df, keys=codes)
    fs_all_df = pd.concat(fs_all_df, keys=codes)

    fs_df.to_pickle(f"data/dart_{year}_df.pkl")
    fs_all_df.to_pickle(f"data/dart_all_{year}_df.pkl")
    print(f"fs_net: {fs_no}, fs_all: {fs_all_no} companiess saved in total of {len(codes)} ones")

    return fs_df, fs_all_df


if __name__ == "__main__":
    load_dotenv(verbose=True)
    api_key = os.getenv("dart_key")
    dart = OpenDartReader(api_key)

    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 31)

    df = get_marcap_period(start=start, end=end)
    df = df.loc[df["Code"].str.endswith("0")]
    df = df.loc[df["Market"] == "KOSPI"]
    print(f"종목수(KOSPI, 보통주): {len(df):,}")

    marcap_df = list()
    for dt in pd.date_range(start=start, end=end, freq="AS"):
        data = df.loc[str(dt.year)].copy()
        vol_quantile = data["Volume"].quantile(q=0.3, interpolation="linear")
        data = data.loc[data["Volume"] > vol_quantile]
        marcap_df.append(data)
        print(f"{dt.year} : {len(data):,}")
    marcap_df = pd.concat(marcap_df)
    marcap_df = marcap_df.droplevel(1).rename_axis("Date")
    print(marcap_df.head())

    fs, fs_all, keys = list(), list(), list()
    for dt in pd.date_range(start=start, end=end, freq="AS"):
        data = df.loc[str(dt.year)].copy()
        codes = data["Code"].values
        try:
            fs_df = pd.read_pickle(f"data/dart_{dt.year}_df.pkl")
            fs_all_df = pd.read_pickle(f"data/dart_all_{dt.year}_df.pkl")
        except FileNotFoundError:
            fs_df, fs_all_df = get_dart_fs(codes=codes, year=dt.year)

        fs.append(fs_df)
        fs_all.append(fs_all_df)
        keys.append(dt.year)

    fs = pd.concat(fs, keys=keys)
    fs_all = pd.concat(fs_all, keys=keys)

    fs_all = (
        fs_all.reset_index()
        .rename(columns={"level_0": "연도", "level_1": "종목"})
        .drop("level_2", axis=1)
        .set_index("연도")
    )
    fs_all.rename(columns=fs_col_names, inplace=True)

    fs_annual = fs_all.loc["2020"]
    fs_annual = fs_annual.loc[
        (fs_annual["재무제표구분"] == "BS")
        | (fs_annual["재무제표구분"] == "IS")
        | (fs_annual["재무제표구분"] == "CF")
    ]
    fs_code = fs_annual.loc[fs_annual["종목"] == "005930"][["계정명", "당기금액"]].set_index("계정명")

    accounts_name = {
        "asset": "자산총계",
        "liability": "부채총계",
        "equity": "자본총계",
        "capital": "자본금",
        "sales": "수익(매출액)",
        "gross_profit": "매출총이익",
        "operating_income": "영업이익",
        "net_income": "당기순이익(손실)",
        "cash_flow_operating": "영업활동 현금흐름",
        "cash_flow_financing": "재무활동 현금흐름",
        "cash_flow_investing": "투자활동 현금흐름",
        "EPS": "기본주당이익(손실)",
    }

    current = dict()
    invest_info = dict()
    for key, value in accounts_name.items():
        current[key] = int(fs_code.at[value, "당기금액"])
        invest_info[value] = current[key]
        print(f"{value}: {current[key]:,}")

    marcap_annual = marcap_df.loc["2020"].set_index("Code")
    marcap_annual.info()

    invest_info["종목"] = "005930"
    invest_info["종가"] = int(marcap_annual.at["005930", "Close"])
    invest_info["거래량"] = int(marcap_annual.at["005930", "Volume"])
    invest_info["거래대금"] = int(marcap_annual.at["005930", "Amount"])
    invest_info["시가총액(백만원)"] = int(marcap_annual.at["005930", "Marcap"])
    invest_info["상장주식수"] = int(marcap_annual.at["005930", "Stocks"])
    invest_info["PER"] = invest_info["종가"] / current["EPS"]
    invest_info["PBR"] = invest_info["시가총액(백만원)"] / current["equity"]
    invest_info["PSR"] = invest_info["시가총액(백만원)"] / current["sales"]
    invest_info["PCR"] = invest_info["시가총액(백만원)"] / current["cash_flow_operating"]
    invest_info["ROE"] = current["net_income"] / current["equity"]
    pprint(invest_info)

    # fs_code.to_csv("data/ss.csv", encoding="utf-8-sig")
