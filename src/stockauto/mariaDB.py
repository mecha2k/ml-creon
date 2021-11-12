import pymysql
import os

from dotenv import load_dotenv


load_dotenv(verbose=True)

conn = pymysql.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWD"),
    db=os.getenv("DB_NAME"),
    port=int(os.getenv("DB_PORT")),
    charset="utf8",
)

cursor = conn.cursor()
cursor.execute("select version();")
result = cursor.fetchone()

print(f"Database version : {result}")

conn.close()
