import pymysql
import os, re
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime, timedelta


class MarketDB:
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
        self.codes = dict()
        self.getCompanyInfo()

    def __del__(self):
        self.conn.close()

    def getCompanyInfo(self):
        sql = "SELECT * FROM company"
        companyInfo = pd.read_sql(sql, self.conn)
        for idx in range(len(companyInfo)):
            self.codes[companyInfo["code"].values[idx]] = companyInfo["name"].values[idx]

    def getDailyPrice(self, code, start_date=None, end_date=None):
        if start_date is None:
            one_year_ago = datetime.today() - timedelta(days=365)
            start_date = one_year_ago.strftime("%Y-%m-%d")
            print(f"start_date is initialized to '{start_date}'")
        else:
            start_lst = re.split("\D+", start_date)
            if start_lst[0] == "":
                start_lst = start_lst[1:]
            start_year = int(start_lst[0])
            start_month = int(start_lst[1])
            start_day = int(start_lst[2])
            if start_year < 1900 or start_year > 2200:
                print(f"ValueError: start_year({start_year:d}) is wrong.")
                return
            if start_month < 1 or start_month > 12:
                print(f"ValueError: start_month({start_month:d}) is wrong.")
                return
            if start_day < 1 or start_day > 31:
                print(f"ValueError: start_day({start_day:d}) is wrong.")
                return
            start_date = f"{start_year:04d}-{start_month:02d}-{start_day:02d}"

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
            print(f"end_date is initialized to '{end_date}'")
        else:
            end_lst = re.split("\D+", end_date)
            if end_lst[0] == "":
                end_lst = end_lst[1:]
            end_year = int(end_lst[0])
            end_month = int(end_lst[1])
            end_day = int(end_lst[2])
            if end_year < 1800 or end_year > 2200:
                print(f"ValueError: end_year({end_year:d}) is wrong.")
                return
            if end_month < 1 or end_month > 12:
                print(f"ValueError: end_month({end_month:d}) is wrong.")
                return
            if end_day < 1 or end_day > 31:
                print(f"ValueError: end_day({end_day:d}) is wrong.")
                return
            end_date = f"{end_year:04d}-{end_month:02d}-{end_day:02d}"

        codes_keys = list(self.codes.keys())
        codes_values = list(self.codes.values())
        if code in codes_keys:
            pass
        elif code in codes_values:
            idx = codes_values.index(code)
            code = codes_keys[idx]
        else:
            print(f"ValueError: Code({code}) doesn't exist.")
        sql = (
            f"SELECT * FROM price WHERE code = '{code}'"
            f" and date >= '{start_date}' and date <= '{end_date}'"
        )
        df = pd.read_sql(sql, self.conn)
        df.index = df["date"]
        return df


if __name__ == "__main__":
    market_db = MarketDB()
    data = market_db.getDailyPrice("삼성전자", "2010-01-24", "2021-02-28")
    print(data)
