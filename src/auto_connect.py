from pywinauto import application
from dotenv import load_dotenv
import time
import os

load_dotenv(verbose=True)
user_id = os.getenv("creon_id")
user_passwd = os.getenv("creon_pwd")
cert_passwd = os.getenv("creon_cert")
print(user_id)
print(user_passwd)
print(cert_passwd)

os.system("taskkill /IM ncStarter* /F /T")
os.system("taskkill /IM CpStart* /F /T")
os.system("taskkill /IM DibServer* /F /T")
os.system("wmic process where \"name like '%ncStarter%'\" call terminate")
os.system("wmic process where \"name like '%CpStart%'\" call terminate")
os.system("wmic process where \"name like '%DibServer%'\" call terminate")
time.sleep(5)

app = application.Application()

app_arg = "C:/ProgramData/CREON/STARTER/coStarter.exe /prj:cp "
app_arg += f"/id:{user_id} /pwd:{user_passwd} /pwdcert:{cert_passwd} /autostart"
app.start(app_arg)
time.sleep(30)