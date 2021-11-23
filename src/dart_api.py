import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import seaborn as sns
import OpenDartReader
import FinanceDataReader as fdr
import os

from marcap import marcap_data
from datetime import datetime
from dotenv import load_dotenv
from icecream import ic


pd.set_option("display.float_format", lambda x: "%.1f" % x)
pd.set_option("max_columns", None)

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
rep_codes = {"1분기": "11013", "반기": "11012", "3분기": "11014", "연간": "11011"}
idx = pd.MultiIndex


def init_dart():
    load_dotenv(verbose=True)
    api_key = os.getenv("dart_key")
    return OpenDartReader(api_key)


def get_corp_fs(dart, corp_code, start, end=None):
    if end is None:
        end = datetime.now()

    corp_fs, keys = list(), list()
    for dt in pd.date_range(start=start, end=end, freq="AS"):
        fs = dart.finstate(corp_code, dt.year)
        if fs is None:
            break
        fs.rename(columns=fs_col_names, inplace=True)
        fs = fs.loc[(fs["개별/연결구분"] == "CFS")]
        fs = fs[["계정명", "당기금액"]].set_index("계정명")
        keys.append(dt)
        corp_fs.append(fs)

    if corp_fs:
        return pd.concat(corp_fs, keys=keys)
    else:
        return None


def get_corp_fs_all(dart, corp_code, start, end=None):
    if end is None:
        end = start
    elif end.year >= datetime.now().year:
        end = datetime(end.year - 1, end.month, end.day)

    corp_fs, corp_fs_all, keys = list(), list(), list()
    for dt in pd.date_range(start=start, end=end, freq="AS"):
        fs = dart.finstate(corp_code, dt.year)
        if fs is None:
            break
        fs.rename(columns=fs_col_names, inplace=True)
        fs = fs.loc[(fs["개별/연결구분"] == "CFS")]
        fs = fs[["계정명", "당기금액"]].set_index("계정명")

        fs_all = dart.finstate_all(corp_code, dt.year)
        if fs_all is None:
            break
        fs_all.rename(columns=fs_col_names, inplace=True)
        fs_all = fs_all.loc[
            (fs_all["재무제표구분"] == "BS") | (fs_all["재무제표구분"] == "CIS") | (fs_all["재무제표구분"] == "IS")
        ]
        fs_all = fs_all[["계정명", "당기금액"]].set_index("계정명")

        keys.append(dt)
        corp_fs.append(fs)
        corp_fs_all.append(fs_all)

    result = None
    if corp_fs and corp_fs_all:
        result = (pd.concat(corp_fs, keys=keys), pd.concat(corp_fs_all, keys=keys))
    return result


# def get_corp_fs_all(dart, corp_code, start, end):
#     if end.year >= datetime.now().year:
#         end = datetime(end.year - 1, end.month, end.day)
#
#     corp_fs, corp_fs_all = [], []
#     for t in pd.date_range(start=start, end=end, freq="A"):
#         rep_times = {
#             "1분기": datetime(t.year, 3, 31),
#             "반기": datetime(t.year, 6, 30),
#             "3분기": datetime(t.year, 9, 30),
#             "연간": datetime(t.year, 12, 31),
#         }
#
#         fs_brf, fs_all = [], []
#         for key, value in rep_codes.items():
#             fs = dart.finstate(corp_code, t.year, reprt_code=value)
#             fs.rename(columns=fs_col_names, inplace=True)
#             fs = fs.loc[(fs["개별/연결구분"] == "CFS")]
#             fs = fs[["계정명", "당기금액"]].set_index("계정명")
#             fs_brf.append(fs)
#
#             fs = dart.finstate_all(corp_code, t.year, reprt_code=value)
#             fs.rename(columns=fs_col_names, inplace=True)
#             fs = fs.loc[(fs["재무제표구분"] == "BS") | (fs["재무제표구분"] == "CIS") | (fs["재무제표구분"] == "IS")]
#             fs = fs[["계정명", "당기금액"]].set_index("계정명")
#             fs_all.append(fs)
#         y_fs_brf = pd.concat(fs_brf, keys=rep_times.values(), names=["시간", "계정명"])
#         y_fs_all = pd.concat(fs_all, keys=rep_times.values(), names=["시간", "계정명"])
#         corp_fs.append(y_fs_brf)
#         corp_fs_all.append(y_fs_all)
#
#     return pd.concat(corp_fs), pd.concat(corp_fs_all)


if __name__ == "__main__":
    dart = init_dart()
    corp = dart.corp_codes

    start = datetime(2000, 1, 1)
    end = datetime(2021, 12, 1)

    fs = get_corp_fs(dart, "005930", start=start)
    print(fs)

    # for dt in pd.date_range(start, end, freq="AS"):
    #     fs = get_corp_fs(dart, "005380", dt)
    #     print(dt, type(fs))

    # corp_fs, corp_fs_all, keys = list(), list(), list()
    # for dt in pd.date_range(start=start, end=end, freq="AS"):
    #     fs = dart.finstate("005930", dt.year)
    #     print(dt)
    #     fs.rename(columns=fs_col_names, inplace=True)
    #     fs = fs.loc[(fs["개별/연결구분"] == "CFS")]
    #     fs = fs[["계정명", "당기금액"]].set_index("계정명")

    # fs_all = dart.finstate_all(corp_code, dt.year)
    # fs_all.rename(columns=fs_col_names, inplace=True)
    # fs_all = fs_all.loc[
    #     (fs_all["재무제표구분"] == "BS") | (fs_all["재무제표구분"] == "CIS") | (fs_all["재무제표구분"] == "IS")
    # ]
    # fs_all = fs_all[["계정명", "당기금액"]].set_index("계정명")

    # keys.append(dt)
    # corp_fs.append(fs)
    # corp_fs_all.append(fs_all)

    # df = pd.concat(corp_fs, keys=keys)
    # print(df)

    # corp_list = {"삼성전자": "005930", "NAVER": "035420", "카카오": "035720", "현대차": "005380"}
    # corp_fs, corp_fs_all = [], []
    # for name, code in corp_list.items():
    #     corp_name = dart_corp.loc[dart_corp.stock_code == code, "corp_name"].values[0]
    #     fs, fs_all = get_corp_fs(corp_code=code, start=start, end=end)
    #     corp_fs.append(fs)
    #     corp_fs_all.append(fs_all)
    #     print(f"{corp_name} fs downloaded...")
    # corp_fs = pd.concat(corp_fs, axis=0, keys=corp_list.keys())
    # corp_fs_all = pd.concat(corp_fs_all, axis=0, keys=corp_list.keys())
    # corp_fs.to_pickle("data/corp_fs.pkl")
    # corp_fs_all.to_pickle("data/corp_fs_all.pkl")

    # corp_fs = pd.read_pickle("data/corp_fs.pkl")
    # corp_fs_all = pd.read_pickle("data/corp_fs_all.pkl")
    #
    # ticker = "삼성전자"
    # samsung = [corp_fs.loc[ticker], corp_fs_all.loc[ticker]]
    # samsung = pd.concat(samsung)
    #
    # annual = pd.date_range(start=start, end=end, freq="A")
    # samsung = samsung.loc[annual]
    # # samsung = samsung.loc[annual].reset_index().pivot(index="계정명", columns="시간", values="당기금액")
    # samsung.to_csv("data/fs_samsung.csv", encoding="utf-8-sig")
    #
    # idx = pd.IndexSlice
    # print(samsung.loc[idx["2020", "매출액"], :].values[0][0])
    #
    # account = ["매출액", "영업이익", "법인세차감전 순이익", "당기순이익", "자산총계", "부채총계", "자본총계", "기본주당이익(손실)"]
    # df = samsung.loc[idx["2020", account], :]
    # df = df["당기금액"].str.replace(",", "").astype("int64")
    # df = df.reset_index().drop_duplicates(subset="계정명", keep="last").drop(["시간"], axis=1)
    #
    # br_div_col_names = {
    #     "rcept_no": "접수번호",
    #     "corp_cls": "법인구분",
    #     "corp_code": "고유번호",
    #     "corp_name": "법인명",
    #     "se": "구분",  # 배당
    #     "stock_knd": "주식종류",
    #     "thstrm": "당기",
    #     "frmtrm": "전기",
    #     "lwfr": "전전기",
    # }
    #
    # br_minor_col_names = {
    #     "rcept_no": "접수번호",
    #     "corp_cls": "법인구분",
    #     "corp_code": "고유번호",
    #     "corp_name": "법인명",
    #     "se": "구분",  # 소액주주
    #     "shrholdr_co": "주주수",
    #     "shrholdr_tot_co": "전체주주수",
    #     "shrholdr_rate": "주주비율",
    #     "hold_stock_co": "보유주식수",
    #     "stock_tot_co": "총발행주식수",
    #     "hold_stock_rate": "보유주식비율",
    # }
    #
    # # 사업보고서 (business report) : 배당
    # br = dart.report(corp="005930", key_word="배당", bsns_year="2020", reprt_code="11011")
    # br.rename(columns=br_div_col_names, inplace=True)
    # br.to_csv("data/br_samsung.csv", encoding="utf-8-sig")
    #
    # EPS = int(br.loc[(br["구분"] == "(연결)주당순이익(원)"), "당기"].str.replace(",", ""))
    # TD = int(br.loc[(br["구분"] == "현금배당금총액(백만원)"), "당기"].str.replace(",", ""))
    # DPS = int(
    #     br.loc[(br["구분"] == "주당 현금배당금(원)") & (br["주식종류"] == "보통주"), "당기"].str.replace(",", "")
    # )
    # Yield = float(
    #     br.loc[(br["구분"] == "현금배당수익률(%)") & (br["주식종류"] == "보통주"), "당기"].str.replace(",", "")
    # )
    # print("EPS : ", EPS, TD, DPS, Yield)
    #
    # # 사업보고서 (business report) : 소액주주
    # br = dart.report(corp="005930", key_word="소액주주", bsns_year="2020", reprt_code="11011")
    # br.rename(columns=br_minor_col_names, inplace=True)
    # stock_tot = int(br["총발행주식수"].str.replace(",", ""))
    #
    # accounts = [
    #     {"계정명": "주당순이익", "당기금액": EPS},
    #     {"계정명": "현금배당금총액(백만원)", "당기금액": TD},
    #     {"계정명": "주당현금배당금(원)", "당기금액": DPS},
    #     # {"계정명": "현금배당수익률(%)", "당기금액": Yield},
    #     {"계정명": "총발생주식수", "당기금액": stock_tot},
    # ]
    # df = df.append(accounts, ignore_index=True)
    # print(df.head(20))
    # print(df.dtypes)
    #
    # equity = int(df.loc[df["계정명"] == "자본총계", "당기금액"])
    # liability = int(df.loc[df["계정명"] == "부채총계", "당기금액"])
    # netincome = int(df.loc[df["계정명"] == "당기순이익", "당기금액"])
    # asset = equity + liability
    # stock_tot_1 = int(netincome / EPS)
    # print(f"자산: {int(asset/1E8):,}, 부채: {int(liability/1E8):,}, 자본: {int(equity/1E8):,} (억원)")
    #
    # df = fdr.DataReader(corp_list["삼성전자"], start=start, end=end)
    # df = df.resample(rule="Y").last()
    # close = int(df["Close"]["2020"])
    #
    # PER = close / EPS
    # BPS = int(equity / stock_tot)
    # PBR = close / BPS
    # ROA = PBR / PER
    # print(f"PER: {PER:.2f}, BPS: {BPS:,}, PBR: {PBR:.2f}, ROA: {ROA:.2f}")
