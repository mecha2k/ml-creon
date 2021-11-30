import pymysql
import calendar
import json
import os
import requests
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime
from threading import Timer
from dotenv import load_dotenv


class Database:
    def __init__(self):
        load_dotenv(verbose=True)
        print(os.getenv("DB_NAME"), os.getenv("DB_HOST"))

        self.conn = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWD"),
            db=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT")),
            charset="utf8",
        )
        self.codes = dict()

    def __del__(self):
        self.conn.close()

    def update_comp_info(self):
        sql = "SELECT * FROM company"
        df = pd.read_sql(sql, self.conn)
        for idx in range(len(df)):
            self.codes[df["code"].values[idx]] = df["name"].values[idx]

        with self.conn.cursor() as curs:
            sql = "SELECT max(updatetime) FROM company"
            curs.execute(sql)
            rows = curs.fetchone()
            today = datetime.today().strftime("%Y-%m-%d %H:%M")
            if rows[0] is None or rows[0].strftime("%Y-%m-%d") < today:
                krx = self.read_krx_code()
                krx = krx.fillna("missing")
                for idx in range(len(krx)):
                    code = krx.code.values[idx]
                    name = krx.name.values[idx]
                    sql = (
                        f"INSERT IGNORE INTO company "
                        f"(code, name, field, publicdate, homepage) VALUES "
                        f"('{code}', '{name}', '{krx.field.values[idx]}', '{krx.publicdate.values[idx]}', "
                        f"'{krx.homepage.values[idx]}')"
                    )
                    curs.execute(sql)
                    self.codes[code] = name
                    timenow = datetime.now().strftime("%Y-%m-%d %H:%M")
                    print(
                        f"[{timenow}] #{idx + 1:04d} INSERT IGNORE INTO company "
                        f"VALUES ({code}, {name}, {krx.products.values[idx]}, {krx.homepage.values[idx]}, {today})"
                    )
                self.conn.commit()

    @staticmethod
    def read_krx_code():
        url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        krx = pd.read_html(url, header=0)[0]
        krx = krx.rename(
            columns={
                "종목코드": "code",
                "회사명": "name",
                "업종": "field",
                "주요제품": "products",
                "상장일": "publicdate",
                "대표자명": "representative",
                "홈페이지": "homepage",
                "지역": "region",
            }
        )
        krx["code"] = krx["code"].map("{:06d}".format)
        return krx

    @staticmethod
    def read_naver_sise(code, company, pages_to_fetch):
        try:
            url = f"http://finance.naver.com/item/sise_day.nhn?code={code}"
            headers = {"User-agent": "Mozilla/5.0"}
            html = BeautifulSoup(requests.get(url, headers=headers).text, "lxml")
            pgrr = html.find("td", class_="pgRR")
            if pgrr is None:
                return None
            s = str(pgrr.a["href"]).split("=")
            lastpage = s[-1]
            df = pd.DataFrame()
            pages = min(int(lastpage), pages_to_fetch)
            for page in range(1, pages + 1):
                pg_url = f"{url}&page={page}"
                df = df.append(pd.read_html(requests.get(pg_url, headers=headers).text)[0])
                tmnow = datetime.now().strftime("%Y-%m-%d %H:%M")
                print(
                    f"[{tmnow}] {company} ({code}) : {page:04d}/{pages:04d} pages are downloading...",
                    end="\r",
                )
            df = df.rename(
                columns={
                    "날짜": "date",
                    "종가": "close",
                    "전일비": "diff",
                    "시가": "open",
                    "고가": "high",
                    "저가": "low",
                    "거래량": "volume",
                }
            )
            df["date"] = df["date"].replace(".", "-")
            df = df.dropna()
            df[["close", "diff", "open", "high", "low", "volume"]] = df[
                ["close", "diff", "open", "high", "low", "volume"]
            ].astype(int)
            df = df[["date", "open", "high", "low", "close", "diff", "volume"]]
            print(df.loc[:5, "date"])
        except Exception as e:
            print("Exception occured :", str(e))
            return None
        return df

    def replace_into_db(self, df, num, code, company):
        with self.conn.cursor() as curs:
            for r in df.itertuples():
                sql = f"SELECT id FROM company WHERE code = '{code}'"
                curs.execute(sql)
                codeid = curs.fetchone()[0]
                sql = (
                    f"INSERT IGNORE INTO price (code, date, open, high, low, close, diff, volume) VALUES "
                    f"('{codeid}', '{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, {r.diff}, {r.volume})"
                )
                curs.execute(sql)
            self.conn.commit()
            timenow = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(
                f"[{timenow}] #{num+1:04d} {company} ({code}) : {len(df)} rows > INSERT IGNORE INTO price [OK]"
            )

    def update_daily_price(self, pages_to_fetch):
        for idx, code in enumerate(self.codes):
            df = self.read_naver_sise(code, self.codes[code], pages_to_fetch)
            if df is None:
                continue
            self.replace_into_db(df, idx, code, self.codes[code])

    def execute_daily(self):
        self.update_comp_info()
        try:
            with open("config.json", "r") as in_file:
                config = json.load(in_file)
                pages_to_fetch = config["pages_to_fetch"]
        except FileNotFoundError:
            with open("config.json", "w") as out_file:
                pages_to_fetch = 1
                config = {"pages_to_fetch": pages_to_fetch}
                json.dump(config, out_file)
        self.update_daily_price(pages_to_fetch)

        # tmnow = datetime.now()
        # lastday = calendar.monthrange(tmnow.year, tmnow.month)[1]
        # if tmnow.month == 12 and tmnow.day == lastday:
        #     tmnext = tmnow.replace(year=tmnow.year + 1, month=1, day=1, hour=17, minute=0, second=0)
        # elif tmnow.day == lastday:
        #     tmnext = tmnow.replace(month=tmnow.month + 1, day=1, hour=17, minute=0, second=0)
        # else:
        #     tmnext = tmnow.replace(day=tmnow.day + 1, hour=17, minute=0, second=0)
        # tmdiff = tmnext - tmnow
        # secs = tmdiff.seconds
        # thread = Timer(secs, self.execute_daily)
        # print("Waiting for next update ({}) ... ".format(tmnext.strftime("%Y-%m-%d %H:%M")))
        # thread.start()


if __name__ == "__main__":
    mysqldb = Database()
    # mysqldb.execute_daily()

    # mysqldb.update_comp_info()
    # print(mysqldb.read_krx_code())

    code = "005930"
    pages_to_fetch = 2
    mysqldb.update_comp_info()
    print(mysqldb.codes[code])

    df = mysqldb.read_naver_sise(code, mysqldb.codes[code], pages_to_fetch)
    mysqldb.replace_into_db(df, 100, code, mysqldb.codes[code])
