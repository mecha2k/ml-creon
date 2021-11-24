import pandas as pd
import OpenDartReader
import FinanceDataReader as fdr
import os

from marcap import marcap_data
from datetime import datetime, timedelta
from dotenv import load_dotenv


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

    start = datetime(2015, 1, 1)
    end = datetime(2016, 12, 31)

    try:
        df = pd.read_pickle("data/marcap_period.pkl")
        print(f"data reading from file...{len(df):,}")
    except FileNotFoundError:
        df = get_marcap_period(start=start, end=end)
        df.to_pickle("data/marcap_period.pkl")

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

    fs, fs_all, keys = list(), list(), list()
    for dt in pd.date_range(start=start, end=end, freq="AS"):
        data = df.loc[str(dt.year)].copy()
        codes = data["Code"].values
        try:
            fs_df = pd.read_pickle(f"data/dart_{dt.year}_df.pkl")
            fs_all_df = pd.read_pickle(f"data/dart_all_{dt.year}_df.pkl")
        except FileNotFoundError:
            fs_df, fs_all_df = get_dart_fs(codes=codes, year=dt.year)

        fs_df.rename(columns=fs_col_names, inplace=True)
        # fs_df = fs_df.loc[(fs_df["개별/연결구분"] == "CFS")]
        fs_df = fs_df[["계정명", "당기금액"]].droplevel(1).reset_index()
        fs_df = fs_df.rename({"index": "종목"}, axis=1).set_index("종목")

        fs_all_df.rename(columns=fs_col_names, inplace=True)
        # fs_all_df = fs_all_df.loc[
        #     (fs_all_df["재무제표구분"] == "BS")
        #     | (fs_all_df["재무제표구분"] == "CIS")
        #     | (fs_all_df["재무제표구분"] == "IS")
        # ]
        fs_all_df = fs_all_df[["계정명", "당기금액"]].droplevel(1).reset_index()
        fs_all_df = fs_all_df.rename({"index": "종목"}, axis=1)

        fs.append(fs_df)
        fs_all.append(fs_all_df)
        keys.append(dt.year)

    fs_df = pd.concat(fs, keys=keys)
    fs_all_df = pd.concat(fs_all, keys=keys)
    fs_df = fs_df.droplevel(1).rename_axis("Time")
    fs_all_df = fs_all_df.droplevel(1).rename_axis("Time")

    idx = pd.IndexSlice
    fs_all_df.info()
    # print(fs_all_df.loc["2015"][fs_all_df["종목"] == "005930"])
    print(fs_all_df.head())
    # print(fs_all_df.loc[idx["2015", "005930"], :])
    fs_df.to_csv("data/fs_df.csv", encoding="utf-8-sig")
    fs_all_df.to_csv("data/fs_all_df.csv", encoding="utf-8-sig")
