"""
File name: basics.py

Creation Date: Tue 03 Aug 2021

Description:
    Implement a basic algorithmic trading strategy and testing it over some months
    data.
    Src: https://www.youtube.com/watch?v=tGKQVML_gSY&list=PLMSih0au9SsDWnhClQbJLIYnJW9rgILfx&index=2

"""

# Python Libraries
# -----------------------------------------------------------------------------
import numpy as np
from yahoo_fin.stock_info import get_data
from trade_stat_logger.logger import SimpleLogger

# Local Application Modules
# -----------------------------------------------------------------------------

"""
Strategy: Buy on open on Tuesday, sell on close. On Friday, short on open, cover short on close.
"""

x = get_data(ticker="aapl", start_date="01/01/2018", end_date = "10/01/2018")
print(x.head())
logger = SimpleLogger()

for idx, row in x.iterrows():
    date = idx.to_pydatetime()
    day = date.weekday()
    print(date.weekday())
    
    if day == 1: ## Day is Tuesday
        logger.log(security="aapl", shares=100, share_price=row["open"])
        logger.log(security="aapl", shares=-100, share_price=row["close"])
    elif day == 5:
        logger.log(security="aapl", shares=-100, share_price=row["open"])
        logger.log(security="aapl", shares=100, share_price=row["close"])

logger.graph_statistics()
    




