import logging
import os

import pandas as pd

from quantylab.systrader.xing.env import Env

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Strategy
# DEBUG = True
DEBUG = False
DEMO = True
# DEMO = False
TRANSACTION_TERM = 2  # 2 seconds
PROCESSING_TERM = 2  # 2 seconds
MARKET_WAIT_TERM = 10  # 10 seconds
MAX_TARGET_STOCK_PRICE = 500000
MAX_BUY_PRICE_AGG = 1000000
MAX_BUY_PRICE_DEF = 500000
BUY_UNIT_AGG = 500000
BUY_UNIT_DEF = 100000
TGT_TOP_DIFF = 10
TGT_BOTTOM_DIFF = -3
MIN_PRICE_VOLUME = 10000 * 10000
# Number of Holdings
MAX_NUM_HOLDINGS_AGG = 12
MAX_NUM_HOLDINGS_DEF = 5
# MAX_NUM_HOLDINGS_DEF = 0
# Monitoring Stocks
MAX_STOCKS_MONITOR_ITR = 5 # Each of KOSDAQ and KOSPI
FIVEMIN_INCDEC_RATE = 0.025


# Settings for Server/
SERVER_ADDR = "localhost"
SERVER_PORT = 8000
SERVER_URL = "http://%s:%s" % (SERVER_ADDR, SERVER_PORT)
SERVER_API_URL = "http://%s:%s/api" % (SERVER_ADDR, SERVER_PORT)
SERVER_WS_URL = "ws://%s:%s/ws" % (SERVER_ADDR, SERVER_PORT)


# Settings for Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Settings for Templates
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


# Settings for Static
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_URL = "/static/"


# Settings on Logging
# LOG_DIR = os.path.join(BASE_DIR, "logs")
# file_handler = logging.FileHandler(filename=os.path.join(LOG_DIR, "%s.log" % Env.get_today_str()), encoding='utf-8')
stream_handler = logging.StreamHandler()
# file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(message)s",
    # format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[
        # file_handler,
        stream_handler
    ],
    level=logging.DEBUG
)


# Settings for Data
DATA_DIR = os.path.join(BASE_DIR, "database")


# Date Time Format
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"



XING_RES_BASE = os.path.join(BASE_DIR, 'res')
XING_RES_PATH = "C:\\eBEST\\xingAPI\\Res\\"
