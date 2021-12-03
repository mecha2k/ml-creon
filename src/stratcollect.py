def find_fscore_stocks(df):
    # calculate F-score (9 indices)
    df["매출총이익률"] = df["매출총이익"] / df["매출액"]
    df["매출총이익률(YoY)"] = df["매출총이익률"].pct_change()
    df["자산회전율"] = df["매출액"] / df["자산총계"]
    df["자산회전율(YoY)"] = df["자산회전율"].pct_change()
    df["부채비율(YoY)"] = -df["부채비율"].pct_change()
    df["ROA(YoY)"] = df["ROA"].pct_change()
    df["영업현금-이익"] = df["영업현금흐름"] - df["영업이익"]

    fscoreIndex = {
        "fscore1": "당기순이익",  # 당기순이익 > 0
        "fscore2": "영업현금흐름",  # 영업현금흐름 > 0
        "fscore3": "ROA(YoY)",  # ROA(YoY) > 0
        "fscore4": "영업현금-이익",  # 영업현금흐름 > 당기순이익
        "fscore5": "부채비율(YoY)",  # 부채비율(YoY) < 0
        "fscore6": "매출총이익률(YoY)",  # 매출총이익률(YoY) > 0
        "fscore7": "자산회전율(YoY)",  # 자산회전율(매출/자산)(YoY) > 0
    }
    for key, value in fscoreIndex.items():
        df[key] = 0
        df.loc[df[value] > 0, key] = 1
    df["fscore_tot"] = 0
    for ind in fscoreIndex.keys():
        df["fscore_tot"] += df[ind]

    return df.sort_values(by=["fscore_tot"], axis=0, ascending=False)


def find_low_value_stocks(df):
    print(df.head())
    df = df.loc[df["매출액"] > 10]
    vol_quantile = df["Volume"].quantile(q=0.3, interpolation="linear")
    df = df.loc[df["Volume"] > vol_quantile]
    equ_quantile = df["자본총계"].quantile(q=0.05, interpolation="linear")
    df = df.loc[df["자본총계"] > equ_quantile]
    df = df.loc[df["자본총계"] > df["자본금"]]

    df = df.loc[df["PBRc"] > 0.5]
    df = df.loc[df["PCRc"] > 2.0]
    df = df.loc[df["PERc"] > 5.0]
    df = df.loc[df["PEGc"] > 0.0]

    df["PBR_rank"] = df["PBRc"].rank(ascending=True)
    df["PSR_rank"] = df["PSRc"].rank(ascending=True)
    df["PCR_rank"] = df["PCRc"].rank(ascending=True)
    df["PER_rank"] = df["PERc"].rank(ascending=True)
    df["PEG_rank"] = df["PEGc"].rank(ascending=True)
    df["DIV_rank"] = df["현금배당수익률"].rank(ascending=False)
    df["EV_rank"] = df["EVc"].rank(ascending=False)
    df["rank_tot"] = (
        df["PBR_rank"]
        + df["PSR_rank"]
        + df["PCR_rank"]
        + df["PER_rank"]
        + df["PEG_rank"]
        + df["DIV_rank"]
    )

    return df.sort_values(by=["rank_tot"], axis=0, ascending=True)
