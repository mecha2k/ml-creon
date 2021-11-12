import os
from dotenv import load_dotenv
from creon import Creon

if __name__ == "__main__":
    load_dotenv(verbose=True)
    id_ = os.getenv("creon_id")
    pwd = os.getenv("creon_pwd")
    cert = os.getenv("creon_cert")
    creon_path = "C:/ProgramData/CREON/STARTER/coStarter.exe /prj:cp"

    creon = Creon()
    conn = creon.connect(id_=id_, pwd=pwd, pwdcert=cert, c_path=creon_path)
    if conn is True:
        print("connection established to creonPlus...")

    data = creon.get_chart("005930", n=60)
    print(data)
