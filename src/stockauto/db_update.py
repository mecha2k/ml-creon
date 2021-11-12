import pymysql
import calendar
import time
import json
import os
import requests
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime
from threading import Timer
from dotenv import load_dotenv


class DB_update:
    def __init__(self):
        load_dotenv(verbose=True)
        db_name = os.getenv("MARIADB_NAME")
        db_port = int(os.getenv("MARIADB_PORT"))
        db_passwd = os.getenv("MARIADB_PASSWD")

        self.conn = pymysql.connect(
            host="localhost",
            user="root",
            password=db_passwd,
            db=db_name,
            port=db_port,
            charset="utf8",
        )

        with self.conn.cursor() as cursor:
            sql = """
            CREATE TABLE IF NOT EXISTS company (
                code VARCHAR(20),
                name VARCHAR(40),
                last_update DATE,
                PRIMARY KEY (code))
            """
            cursor.execute(sql)
            sql = """
            CREATE TABLE IF NOT EXISTS price (
                code VARCHAR(20),
                date DATE,
                open BIGINT(20),
                high BIGINT(20),
                low BIGINT(20),
                close BIGINT(20),
                diff BIGINT(20),
                volume BIGINT(20),
                PRIMARY KEY (code, date))
            """
            cursor.execute(sql)
        self.conn.commit()
        self.codes = dict()

    def __del__(self):
        self.conn.close()

    def update_comp_info(self):
        sql = "SELECT * FROM company"
        df = pd.read_sql(sql, self.conn)
        for idx in range(len(df)):
            self.codes[df["code"].values[idx]] = df["name"].values[idx]

        with self.conn.cursor() as curs:
            sql = "SELECT max(last_update) FROM company"
            curs.execute(sql)
            rs = curs.fetchone()
            today = datetime.today().strftime("%Y-%m-%d")
            if rs[0] is None or rs[0].strftime("%Y-%m-%d") < today:
                krx = self.read_krx_code()
                for idx in range(len(krx)):
                    code = krx.code.values[idx]
                    name = krx.name.values[idx]
                    sql = (
                        f"REPLACE INTO company (code, name, last_update) "
                        f"VALUES ('{code}', '{name}', '{today}')"
                    )
                    curs.execute(sql)
                    self.codes[code] = name
                    tmnow = datetime.now().strftime("%Y-%m-%d %H:%M")
                    print(
                        f"[{tmnow}] #{idx + 1:04d} REPLACE INTO company "
                        f"VALUES ({code}, {name}, {today})"
                    )
                self.conn.commit()
                print("")

    @staticmethod
    def read_krx_code():
        url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        krx = pd.read_html(url, header=0)[0]
        krx = krx[["종목코드", "회사명"]]
        krx = krx.rename(columns={"종목코드": "code", "회사명": "name"})
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
        except Exception as e:
            print("Exception occured :", str(e))
            return None
        return df

    def replace_into_db(self, df, num, code, company):
        with self.conn.cursor() as curs:
            for r in df.itertuples():
                sql = (
                    f"REPLACE INTO price VALUES ('{code}', "
                    f"'{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, {r.diff}, {r.volume})"
                )
                curs.execute(sql)
            self.conn.commit()
            cur_time = datetime.now().strftime("%Y-%m-%d" " %H:%M")
            print(
                f"[{cur_time}] #{num+1:04d} {company} ({code}) : {len(df)} rows > REPLACE INTO price [OK]"
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

        tmnow = datetime.now()
        lastday = calendar.monthrange(tmnow.year, tmnow.month)[1]
        if tmnow.month == 12 and tmnow.day == lastday:
            tmnext = tmnow.replace(year=tmnow.year + 1, month=1, day=1, hour=17, minute=0, second=0)
        elif tmnow.day == lastday:
            tmnext = tmnow.replace(month=tmnow.month + 1, day=1, hour=17, minute=0, second=0)
        else:
            tmnext = tmnow.replace(day=tmnow.day + 1, hour=17, minute=0, second=0)
        tmdiff = tmnext - tmnow
        secs = tmdiff.seconds
        thread = Timer(secs, self.execute_daily)
        print("Waiting for next update ({}) ... ".format(tmnext.strftime("%Y-%m-%d %H:%M")))
        thread.start()


if __name__ == "__main__":
    db_update = DB_update()
    db_update.execute_daily()
