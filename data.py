import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import datetime as dt
from pandas_datareader.yahoo.options import Options as YahooOptions

" this class give the choice to the user to import the stocks"

" this class import data of 5 y at the time of using it ( if you launch it in 2023 it will import data from 2018 to 2023)"


class my_data:
    def __init__(self, *ticker):

        """ ticker is dynamic"""

        self.donnee = pd.DataFrame()

        self.ticker = ticker

        self.log_returns = pd.DataFrame()

        self.stdev = pd.DataFrame()
        " import the stock price from yahoo finance"
        for t in ticker:
            self.donnee[t] = wb.DataReader(t, data_source='yahoo',
                                           start=(dt.date.today() - dt.timedelta(days=5 * 365.24 + 2)).strftime(
                                               "%d-%m-%Y"), end=dt.date.today().strftime("%d-%m-%Y"))['Adj Close']

    "calculate the log_return of each stocks"

    def log_return(self):

        self.log_returns = np.log(1 + self.donnee.pct_change())

        return (self.log_returns)

    "calculate the annual volatility of returns"

    def std_return(self):

        self.stdev = self.donnee.pct_change().std() * 250 ** 0.5

        return (self.stdev)

    def __select_min_call(self, df_call_market, k, a, m, j):
        """ private function used to help find call
                df_call_market is dataframe of the call market
                this function return the minimum call value
                a: year
                m: month
                j : day
                k : strike

        """
        return ((df_call_market[(df_call_market.index.get_level_values(0) == k) &
                                (df_call_market.index.get_level_values(1) == dt.datetime.strptime(
                                    str(a) + '-' + str(m) + '-' + str(j), '%Y-%m-%d'))].iloc[0]['Last']))

    def __get_data(self, year, month, day, tickers, strike):
        """ this is a private function

            this function gives the price of one call knowing its year, month,day and strike

           the dataframe manipulated is multiindex ( jutification of using ( get_level_values(0)=strike)

           it returns the cheapest call price at (year-month-day)
           """
        # this function the price of one call knowing its year, month,day and strike
        # the dataframe manipulated is multiindex ( jutification of using ( get_level_values(0)=strike)

        C_frame = YahooOptions(tickers).get_call_data(year=year, month=month)

        C_frame = self.__select_min_call(df_call_market=C_frame, k=strike, a=year, m=month, j=day)

        " to construct Dataframe, we fill the  empty value with zero"
        if not C_frame.size > 0:
            C_frame = np.append(C_frame, 0)
        return (C_frame)

    def market_call_price(self, annee, mois, jour, tickers, K):
        """ market call price is an extension of get_call_data applied to a group of ticker"""
        """ columns are tickers
          values are the market price ( Last  operation price)
            the program user must only call the market_call_price """
        my_dict = {}

        my_dict[tickers] = [self.__get_data(annee, mois, jour, tickers, K)]

        return (pd.DataFrame.from_dict(my_dict))

    """ these functions may return only 0  because there's no one offering this product in the market
    
    
    # the function vizualize_call is just a function to see the call possible in this month for a give ticker """

    def vizualize_calls(self, yearr, montth, tickerr):

        "the function will help the costumers look at offers available in the market"

        return (YahooOptions(tickerr).get_call_data(year=yearr, month=montth))
