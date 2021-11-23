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


if __name__ == "__main__":
    load_dotenv(verbose=True)
    api_key = os.getenv("dart_key")
    dart = OpenDartReader(api_key)

    start = datetime(2000, 1, 1)
    end = datetime(2021, 12, 31)

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

    for dt in pd.date_range(start=start, end=end, freq="AS"):
        fs = dart.finstate("005380", dt.year)
        # fs.rename(columns=fs_col_names, inplace=True)
        # fs = fs.loc[(fs["개별/연결구분"] == "CFS")]
        # fs = fs[["계정명", "당기금액"]].set_index("계정명")
        # keys.append(dt)
        # corp_fs.append(fs)
        print(dt, fs)

    # dart = dart_api.init_dart()
    #
    # fiscal_df = list()
    # for dt in pd.date_range(start=start, end=end, freq="AS"):
    #     fs_df, fs_all_df = list(), list()
    #     for code in marcap_df.loc[str(dt.year)]["Code"].values:
    #         fs, fs_all = dart_api.get_corp_fs(dart, code, start, end)
    #         print(code)
