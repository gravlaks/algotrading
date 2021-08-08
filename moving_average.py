"""
File name: moving_average.py

Creation Date: Tue 03 Aug 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------
from yahoo_fin.stock_info import get_data
import trade_stat_logger

# Local Application Modules
# -----------------------------------------------------------------------------

""" 
Investment strategy: Whenever ndays_momentum is lower than resistance,
                        buy stocks
"""


def compute_performance_MA(ticker, ndays_momentum,ndays_resistance, 
        start_date, end_date, bandwidth=1.05, threshold_ratio=1.3):

    
    assert(ndays_momentum<ndays_resistance), "ndays_m must be less than ndays_r"

    logger = trade_stat_logger.SimpleLogger()
    data = get_data(ticker, start_date=start_date, end_date=end_date)
    open_data = data["open"]

    ma_resistance = open_data.rolling(ndays_resistance).mean()
    ma_momentum = open_data.rolling(ndays_momentum).mean()

    sliced_data = data[ndays_resistance:]
    ma_resistance = ma_resistance[ndays_resistance:]
    ma_momentum = ma_momentum[ndays_resistance:]

    mom_prev_below_res = False
    position_size = 0 # Cap the number of stocks
    position_size_cap = 2000
    for i in range(len(ma_momentum)):
        mom = ma_momentum.iloc[i]
        res = ma_resistance.iloc[i]
        share_price = sliced_data.iloc[i]["open"]

        if (mom*bandwidth>res and mom_prev_below_res) and (position_size<position_size_cap):
            logger.log(security=ticker, shares = 50, share_price=share_price)
            position_size +=50
            mom_prev_below_res = False
        elif (mom*threshold_ratio>res and mom_prev_below_res) and (position_size>0):
            logger.log(security=ticker, shares = -50, share_price=share_price)
            mom_prev_below_res = False
            position_size -=50
        elif mom<res and position_size>0:
            logger.log(security=ticker, shares = -50, share_price=share_price)
            position_size -=50
            mom_prev_below_res = True
        elif mom<res:
            mom_prev_below_res = True

            
    return logger

        

logger = compute_performance_MA(ticker="aapl", ndays_momentum=20, ndays_resistance=100,
        start_date="01/01/2015", end_date="01/01/2020")
logger.graph_statistics()
