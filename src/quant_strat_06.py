import numpy as np
import pandas as pd
import time

from datetime import datetime
from quantstrat import QuantStrat

import stratcollect


if __name__ == "__main__":
    stock_no = 10
    start = datetime(2018, 5, 1)
    qstrat = QuantStrat(stock_no=stock_no, start=start)
    print(f"start : {start}, stock_no : {stock_no}")

    stime = time.time()
    # qstrat.update_investing_data()
    qstrat.get_stocks_from_strategy(stratcollect.find_low_value_stocks)
    # qstrat.optimize_stocks_from_MPT(weight=0.02)
    qstrat.get_investing_yields()
    # qstrat.plot_stock_annual_returns()
    # qstrat.quantstats_reports()

    print(f"\nexecution time elapsed (sec) : {time.time()-stime}")
