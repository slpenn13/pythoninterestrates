''' Common pricing methods corresponding to Interest rate Instruments '''
# import datetime as dt
#from collections import OrderedDict
# import json
# import os
# import scipy as sci
import numpy as np
# import pandas as pd
# import interest_rate_base as intbase
import interest_rate_dates as intdate
import interest_rate_discount as intdisc
# import curve_constructor as cb
# import pyfi_filter as fi_filter


def h_i(tau, t):
    ''' calculates h_prime '''
    if t < 0:
        raise ValueError("Negative Time")

    return tau*t + 0.5*max(tau, t)*min(tau, t)**2 - (1/6.)*min(tau, t)**3

class discount_calculator_lorimier(intdisc.discount_calculator):
    ''' adjustment of discount calculator to account for lorimier specifics '''
    def __init__(self, rates, data_type=0, dates=None, origin=None, dbg=False):
        ''' constructor for discount_calculator_lorimier
        rates:  vector of PRICES
        data_type: must be PRICE (float64)
        dates:  dictionary of assumptions
            paramaters: (element of dates) must include alpha and beta vector

        '''
        if dates and isinstance(dates, dict):
            self.options = dates.copy()
            if "parameters" in self.options.keys():
                if "alpha" in self.options['parameters'].keys() and\
                        "beta" in self.options['parameters'].keys() and\
                        isinstance(self.options['parameters']['beta'], (np.ndarray, list)):
                    if 'dates' not in self.options['parameters'].keys() or\
                            not isinstance(self.options['parameters']['dates'], dict):
                        raise ValueError("Faulty date specification")
                else:
                    raise ValueError("Parameters must include alpha and beta as parameters")
            else:
                raise ValueError("Parameters must a element of options dictionary")
        else:
            raise ValueError("Options must be a dictionary")

        super().__init__(rates, data_type=data_type, dates=self.options['parameters']['dates'],
                         origin=1, dbg=dbg)

    def f0(self, t, dbg=False):
        ''' calculates f0 @ time t based on provided beta '''
        if 'parameters' in self.options.keys() and 'beta' in self.options['parameters'].keys():
            res = self.options['parameters']['beta'][0]*t

            for tau, beta in zip(self.options['parameters']['tau'],
                                 self.options['parameters']['beta'][1:]):
                hi = h_i(tau, t)
                res = res + beta*hi
                if dbg or self.debug:
                    print(t, beta, hi, tau)
        else:
            raise ValueError("Missing f0 elements")

        return res / t

    def interpolate_zero(self, maturity: float):
        ''' calculates zero given maturity '''
        # print("Interpolate: %f " % maturity)
        yt = -0.01*maturity*self.f0(maturity)
        return np.exp(yt)

    def forward(self, loc, prev_loc):
        ''' Calculates forward rate based on f0 '''
        ti = self.matrix.loc[loc, 'maturity']
        prev_ti = self.matrix.loc[prev_loc, 'maturity']

        weights = np.zeros(2)
        weights[0] = prev_ti / ti
        mat = self.matrix.loc[loc, 'maturity']
        f0 = self.f0(mat)
        forward = (f0*ti - self.matrix.loc[prev_loc, 'yield_hat']*prev_ti)/(ti - prev_ti)

        return forward, f0

    def init_prices(self, prices):
        ''' results in the case of yields '''
        mult = 100.0

        for i in np.arange(0, self.matrix.shape[0]):
            sched = self.schedule[i]
            if i == 0:
                self.matrix.loc[sched, 'zero'] = 1.0
                schedprev = 0
            else:
                mat = intdate.calc_bdte_diff(
                    self.schedule[i], self.date_spec, self.schedule[0])

                self.matrix.loc[sched, 'yield'] = -mult*np.log(prices[i-1]/mult)/mat

                self.matrix.loc[sched, 'maturity'] = mat

                self.matrix.loc[sched, 'date_diff'] = intdate.calc_bdte_diff(
                    self.schedule[i], self.date_spec, self.schedule[i-1])

                if i > 1:
                    self.matrix.loc[sched, ['forward', 'yield_hat']] = self.forward(
                        sched, schedprev)
                else:
                    self.matrix.loc[sched, 'yield_hat'] = self.f0(mat)
                    self.matrix.loc[sched, 'forward'] = self.matrix.loc[sched, 'yield_hat']

                self.matrix.loc[sched, 'zero'] = np.exp(
                    -self.matrix.loc[sched, 'yield_hat']*mat/mult)

                schedprev = sched
