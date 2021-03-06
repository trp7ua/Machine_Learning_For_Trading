"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
  # this is the function the autograder will call to test your code
  # TODO: Your code here

  orders = pd.read_csv(orders_file).sort('Date')
  sym=orders.Symbol.unique()

  start_date =  orders['Date'].iloc[0]
  end_date = orders['Date'].iloc[-1]
  prices_all = get_data(sym.tolist(),pd.date_range(start_date, end_date))
  
  port= dict(zip(sym.tolist(),[0]*len(sym)))
  portList=[]  
  cash = start_val
  leverage= 0.0

  for index, price in prices_all.iterrows():
	  portVal=cash
	  for key,value in port.iteritems():
		  portVal=  portVal+prices_all.ix[index][key]*value
	  portList.append(portVal)

	  if str(index.date()) in orders['Date'].values and str(index.date())!=str(dt.date(2011,6,15)):
	  	for idx, odr in orders[orders['Date']==str(index.date())].iterrows():
	  			num=0
	  			den=0
	  			if odr.Order=='BUY':
	  				port[odr.Symbol]+=odr.Shares
	  				cash-=prices_all.ix[odr.Date][odr.Symbol]*odr.Shares
	  				for key,value in port.iteritems():
	  					num=num+abs(prices_all.ix[odr.Date][key]*value)
	  					den=den+prices_all.ix[odr.Date][key]*value
	  				newLeverage=num/(den+cash)
	  				if newLeverage <= 3.0:
	  					leverage=newLeverage
	  				else:
	  					port[odr.Symbol]-=odr.Shares
	  					cash+=prices_all.ix[odr.Date][odr.Symbol]*odr.Shares							
	  			else:
	  				port[odr.Symbol]-= odr.Shares
	  				cash+=prices_all.ix[odr.Date][odr.Symbol]*odr.Shares
	  				for key,value in port.iteritems():
	  					num=num+abs(prices_all.ix[odr.Date][key]*value)
	  					den=den+prices_all.ix[odr.Date][key]*value
	  				newLeverage=num/(den+cash)
	  				if newLeverage <= 3.0:
	  					leverage=newLeverage
	  				else:
	  					port[odr.Symbol]+=odr.Shares
	  					cash-=prices_all.ix[odr.Date][odr.Symbol]*odr.Shares
  return pd.DataFrame(data=portList,columns=['Daily Value'],index=prices_all.index)

  # In the template, instead of computing the value of the portfolio, we just
  # read in the value of IBM over 6 months
  # start_date = dt.datetime(2008,1,1)
  # end_date = dt.datetime(2008,6,1)
  # portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
  # portvals = portvals[['IBM']]  # remove SPY


def test_code():
  # this is a helper function you can use to test your code
  # note that during autograding his function will not be called.
  # Define input parameters

  of = "./orders/orders3.csv"
  sv = 1000000

  # Process orders
  portvals = compute_portvals(orders_file = of, start_val = sv)

  if isinstance(portvals, pd.DataFrame):
	 portvals = portvals[portvals.columns[0]] # just get the first column
  else:
	"warning, code did not return a DataFrame"
   
  # Get portfolio stats
  # Here we just fake the data. you should use your code from previous assignments.
  start_date = dt.datetime(2008,1,1)
  end_date = dt.datetime(2008,6,1)
  cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
  cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

  # Compare portfolio against $SPX

  print "Date Range: {} to {}".format(start_date, end_date)
  print
  print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
  print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
  print
  print "Cumulative Return of Fund: {}".format(cum_ret)
  print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
  print
  print "Standard Deviation of Fund: {}".format(std_daily_ret)
  print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
  print
  print "Average Daily Return of Fund: {}".format(avg_daily_ret)
  print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
  print
  print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
  test_code()
