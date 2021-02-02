import numpy as np
import pandas as pd

class my_payoff:
    def __init__(self, ST, K):  # ST is a dataframe given by Black and scholes formula

        self.K = K

        self.ST = ST

    def european_call(self):
        return (np.maximum(self.ST[-1] - self.K, 0))

    def european_put(self):
        return (np.maximum(self.K - self.ST[-1], 0))

    def Knock_in_call(self, barriere):
        return (0 if (float(self.ST[-1].max()) <= barriere) else np.maximum(self.ST[-1] - self.K, 0))

    def Knock_in_put(self, barriere):
        return (0 if (float(self.ST[-1].max()) <= barriere) else np.maximum(self.K - self.ST[-1], 0))

    def Knock_out_call(self, barriere):
        return (0 if (float(self.ST[-1].max()) >= barriere) else np.maximum(self.ST[-1] - self.K, 0))

    def Knock_out_put(self, barriere):
        return (0 if (float(self.ST[-1].max()) >= barriere) else np.maximum(self.K - self.ST[-1], 0))

    def asian_call(self):
        return (np.maximum(self.ST.mean(axis=1) - self.K, 0))







