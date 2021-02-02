import pandas as pd
import numpy as np
from scipy.stats import norm
import math


class my_pricing:
    def __init__(self, S, K, r, stdev, T):
        """ S: The value of the underlying asset given by data.py

            ST: is the value of the asset at maturity ( each stock at maturity)

            K : The value of the Strike

            r : risk free

            stdev : standard deviation ( volatility

            T : maturity
        """

        self.S = S

        ST = S.iloc[-1]

        self.ST = ST

        self.K = K

        self.r = r

        self.stdev = stdev

        self.T = T

        self.d1 = (np.log(self.ST / self.K) + (self.r + self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

        self.d2 = (np.log(self.ST / self.K) + (self.r - self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

    def Call_BSM(self):
        """ this is the closed formula of pricing the european call"""

        return (float(self.ST * norm.cdf(self.d1) - self.K * np.exp(- self.r * self.T) * norm.cdf(self.d2)))

    def Put_BSM(self):

        """  return: the closed formula to price the european put """

        return (float(- self.ST * norm.cdf(- self.d1) + self.K * np.exp(- self.r * self.T) * norm.cdf(- self.d2)))

    def delta_call(self):
        """ give the delta value of european call"""
        return (norm.cdf(self.d1))

    def delta_put(self):
        """ give the delta value of european put """
        return (-norm.cdf(-self.d1))

    def gamma(self):
        """ give the gamma value of european option"""
        return (norm.pdf(self.d1) / (self.stdev * np.sqrt(self.T)))  # density proba function

    def vega(self):
        """ give the delta value of european option"""
        return (self.ST * norm.pdf(self.d1) * np.sqrt(self.T))

    def Monte_carlo(self, iterations):
        """ this is the monte carlo simulation
            we generate random stock values
        """
        t_intervals = 250 * self.T

        Z = np.random.standard_normal((t_intervals + 1, iterations))

        St = np.zeros((t_intervals + 1, iterations))

        St[0]= self.ST

        delta_t = self.T / t_intervals
        for t in range(1, t_intervals + 1):
            St[t] = St[t - 1] * np.exp((float(self.r) - 0.5 * float(self.stdev) ** 2) * delta_t + float(self.stdev) * np.sqrt(delta_t) * Z[t])
        return (St)

    def pricing_montecarlo(self, payoff,iterations):
        """  this is the pricing of the option based on monte carlo simulation
            payoff: is given by the class payoff

        """
        return (np.exp(- self.r * self.T) * np.sum(payoff) / iterations)


    def model_heston(self,v0, kappa, theta, sigma, rho, T, M, I):
        """To account for the correlation between the two stochastic processes, we need to determine
            the Cholesky decomposition of the correlation matrix:
             Before we start simulating the stochastic processes, we generate the whole set of random
            numbers for both processes, looking to use set 0 for the stock and set 1 for the
            volatility process
            For the volatility process modeled by the square-root diffusion process type, we use the
            Euler scheme, taking into account the correlation parameter:
             """
        corr_mat = np.zeros((2, 2))
        corr_mat[0, :] = [1.0, rho]
        corr_mat[1, :] = [rho, 1.0]
        cho_mat = np.linalg.cholesky(corr_mat)
        ran_num = np.random.standard_normal((2, M + 1, I))
        dt = T / M
        v = np.zeros_like(ran_num[0])
        vh = np.zeros_like(v)
        v[0] = v0
        vh[0] = v0
        for t in range(1, M + 1):
            ran = np.dot(cho_mat, ran_num[:, t, :])
            vh[t] = (vh[t - 1] + kappa * (theta - np.maximum(vh[t - 1], 0)) * dt
                     + sigma * np.sqrt(np.maximum(vh[t - 1], 0)) * np.sqrt(dt)
                     * ran[1])
        return (np.maximum(vh, 0))

    def asset_heston(self,S0, T, I, v,rho,M,r):
        """:For the stock process, we also take into account the correlation and use the exact
            Euler scheme for the geometric Brownian motion:"""
        corr_mat = np.zeros((2, 2))
        corr_mat[0, :] = [1.0, rho]
        corr_mat[1, :] = [rho, 1.0]
        cho_mat = np.linalg.cholesky(corr_mat)
        ran_num = np.random.standard_normal((2, M + 1, I))
        dt = T / M
        S = np.zeros_like(ran_num[0])
        S[0] = S0
        for t in range(1, M + 1):
            ran = np.dot(cho_mat, ran_num[:, t, :])
            S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
                                     np.sqrt(v[t]) * ran[0] * np.sqrt(dt))
        return (S)
    """
    def mc_heston(self,option_type, S0, K, T, W1, W2, initial_var, long_term_var, rate_reversion, vol_of_vol, r,
                  num_reps, steps):
        
        #             option_type:    'p' put option 'c' call option
        #             S0:              the spot price of underlying stock
        #             K:              the strike price
        #             T:              the maturity of options
        #             initial_var:    the initial value of variance
        #             long_term_var:  the long term average of price variance
        #             rate_reversion: the mean reversion rate for the variance
        #             vol_of_vol:     the volatility of volatility(the variance of the variance of stock price)
        #             corr:           the correlation between variables W1 and W2
        #             r:              the risk free rate
        #             reps:           the number of repeat for monte carlo simulation
        #             steps:          the number of steps in each simulation
        #             W1,W2 : Brownian motions generated using class Brownian

                    

        delta_t = T / float(steps)

        payoff = 0

        for i in range(num_reps):

            vt = initial_var

            st = S0

            for j in range(steps):
                vt = (np.sqrt(vt) + 0.5 * vol_of_vol * np.sqrt(delta_t) * W1) ** 2 \
                     - rate_reversion * (vt - long_term_var) * delta_t \
                     - 0.25 * vol_of_vol ** 2 * delta_t

                st = st * np.exp((r - 0.5 * vt) * delta_t + np.sqrt(vt * delta_t) * 2)
            if option_type == 'c':

                payoff += max(st - K, 0)

            elif option_type == 'p':

                payoff += max(K - st, 0)

        return (payoff / float(num_reps)) * (exp(-r *T)) 
        """

    def erreur_vol_market(self, sigma, MKT, tickeer):
                #MKT is the market value of the call (  use method market_call_price)

         #the pricing is done manually in order to make easier the use of call and put pricing

         #_d11 , _d22 are private variable
            
        _d11 = (np.log(self.ST[tickeer] / self.K) + (self.r + sigma ** 2 / 2) * self.T) / (sigma * np.sqrt(self.T))

        _d22 = (np.log(self.ST[tickeer] / self.K) + (self.r - sigma ** 2 / 2) * self.T) / (sigma * np.sqrt(self.T))

        # print((self.ST * _d11.apply( lambda x : norm.cdf(x)) - self.K * np.exp(- self.r * self.T) * norm.cdf(_d22)))
        # print((self.ST * _d11.apply(lambda x : norm.cdf(x)) - self.K * np.exp(- self.r * self.T) * _d22.apply(lambda x : norm.cdf(x) - MKT)))
        return (
            (self.ST[tickeer] * norm.cdf(_d11) - self.K * np.exp(- self.r * self.T) * norm.cdf(_d22) - MKT[tickeer]))



    def implied_vol(self, MKT_val, ticker1):

        """in each step we try to reach the market volatility
                      at each time we assume that
                      min_vol : the lowest value of volatility ( 1bp)
                      max_val : the highest value of volatility ( 100%)
                      _N : private variable to calculate the number of iterations
                      ticker1: the stock concerned
                      """

        min_vol = 0.0001

        max_vol = 1

        _N = 1

        tol = 10 ** (-4)

        while _N < 10000:

            local_vol = (min_vol + max_vol) / 2

            if (abs(self.erreur_vol_market(local_vol, MKT_val, ticker1)[0] < tol) or (max_vol - min_vol) / 2 < tol):

                return (local_vol)

            elif (np.sign(self.erreur_vol_market(local_vol, MKT_val)[0]) == np.sign(self.erreur_vol_market())):

                min_vol = local_vol
            else:
                max_vol = local_vol
