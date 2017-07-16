"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
from scipy.optimize import minimize
import math

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
     # add code here to find the allocations
    
    x0 = np.random.random(len(syms))
    x0 /= x0.sum()
    # x0=np.asarray([0.2, 0.2, 0.3, 0.3, 0.0])
    fun = lambda x: -sharp_ratio(prices.values,x)
    cons = ({ 'type': 'eq', 'fun': lambda inputs: 1 - np.sum(inputs) })
    bnds = tuple((0,None) for i in range(len(syms)))
    res = minimize(fun, x0 , method='SLSQP', bounds=bnds, constraints=cons)

    allocs = res.x

    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    priceSPY=prices_SPY.values
    priceSPY /= priceSPY[0]

    price_stocks = prices.values
    price_stocks /= price_stocks[0]
    price_stocks *= allocs

    port_val = pd.DataFrame(price_stocks.sum(axis=1),index=prices.index)
    prices_SPY = pd.DataFrame(priceSPY,index=prices.index)



    # Get daily portfolio value
    # port_val = prices_SPY # add code here to compute daily portfolio values

    cr = port_val.values[-1] -1
    dr = port_val.values
    drShift = np.vstack([dr[0],dr[0:(len(dr)-1)]])
    dr = dr/drShift -1

    adr = dr.mean()
    sddr=dr.std()

    k = math.sqrt(252)

    sr = k*np.mean(dr)/np.std(dr)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.columns=df_temp.columns.get_level_values(0)
        plot_data(df_temp, "Daily portfolio value and SPY", "Date", "Normalized prices")

    return allocs, cr, adr, sddr, sr

def sharp_ratio(prices,alc,sf=252):

    pricesShift = np.vstack([prices[0],prices[0:(len(prices)-1)]])
    prices = prices/pricesShift -1
    prices *= alc

    dr = prices.sum(axis=1)

    adr = dr.mean()
    sddr=dr.std()

    k = math.sqrt(sf)

    sr = k*np.mean(dr)/np.std(dr)

    return sr


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols =  ['IBM', 'X', 'HNZ', 'XOM', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
