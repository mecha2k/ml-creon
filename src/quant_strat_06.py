import numpy as np
import pandas as pd

from datetime import datetime
from quantstrat import QuantStrat

if __name__ == "__main__":
    stock_no = 10
    start = datetime(2012, 5, 1)
    qstrat = QuantStrat(stock_no=stock_no, start=start)
    print(f"start : {start}, stock_no : {stock_no}")

    # stime = time.time()
    # fs_df, creon_df, fdr_df = get_investing_info_data()
    # results = analyze_strategy(stock_no, fdr_df, fs_df, start=start)
    # investing_yields(results)
    # confirm_strategy(start, fdr_df, fs_df)
    # print(f"\nexecution time elapsed (sec) : {time.time()-stime}")
