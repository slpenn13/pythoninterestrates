'''Discount class '''
from enum import IntEnum, unique
import numpy as np
import pandas as pd
import interest_rate_base as intbase
import interest_rate_dates as intdate

@unique
class discount_data(IntEnum):
    ''' Enumeration of discount types'''
    ZERO = 0
    FORWARD = 1
    YIELD = 2


class discount_calculator():
    ''' Class containing construction of discout class + zero calculator + interpolator '''
    def __init__(self, rates, data_type=0, dates=None, origin=None, dbg=False):
        ''' builds pd.DataFrame from inputs '''
        if origin is not None and isinstance(origin, list):
            columns = ['maturity', 'date_diff', 'zero', 'forward', 'yield', 'origin']
        else:
            columns = ['maturity', 'date_diff', 'zero', 'forward', 'yield']

        self.rowcount = len(rates)+1
        self.debug = dbg
        mult = 100.0  # calculate results in percents
        self.matrix = pd.DataFrame(np.zeros([self.rowcount, len(columns)]), columns=columns)

        if isinstance(dates, dict):
            self.date_spec = dates.copy()
        else:
            raise ValueError("discount calculator -- faulty schedule specification")

        if 'start' in self.date_spec.keys():  # SCHED0 -- start + application of rowcount
            self.schedule = intdate.calc_schedule(self.date_spec['start'], (self.rowcount-1),
                                                  options=self.date_spec,
                                                  period=self.date_spec['frequency'])
        elif 'schedule' in self.date_spec.keys():  # SCHED1 -- previously constructed schedule
            self.schedule = sorted(self.date_spec['schedule'])
        else:
            raise ValueError("discount calculator: NO schedule generated")

        if data_type == discount_data.ZERO:
            for i in np.arange(0, self.matrix.shape[0]):
                if i == 0:
                    self.matrix.loc[i, 'zero'] = 1.0
                else:
                    self.matrix.loc[i, 'zero'] = rates[i-1]
                    self.matrix.loc[i, 'maturity'] = intdate.calc_bdte_diff(
                        self.schedule[i], self.date_spec, self.schedule[0])

                    self.matrix.loc[i, 'date_diff'] = intdate.calc_bdte_diff(
                        self.schedule[i], self.date_spec, self.schedule[i-1])
                    self.matrix.loc[i, 'forward'] = intbase.calc_forward_rate_1d(
                        self.matrix.loc[(i-1), 'maturity'], self.matrix.loc[i, 'maturity'],
                        self.matrix.loc[(i-1), 'zero'], self.matrix.loc[i, 'zero'], mult)

                    self.matrix.loc[i, 'yield'] = mult*(-1.)*np.log(self.matrix.loc[i, 'zero'])/\
                        self.matrix.loc[i, 'maturity']
        elif data_type == discount_data.FORWARD:
            for i in np.arange(0, self.matrix.shape[0]):
                if i == 0:
                    self.matrix.loc[i, 'zero'] = 1.0
                else:
                    self.matrix.loc[i, 'forward'] = rates[i-1]
                    self.matrix.loc[i, 'maturity'] = intdate.calc_bdte_diff(
                        self.schedule[i], self.date_spec, self.schedule[0])

                    self.matrix.loc[i, 'date_diff'] = intdate.calc_bdte_diff(
                        self.schedule[i], self.date_spec, self.schedule[i-1])
                    self.matrix.loc[i, 'zero'] = intbase.calc_forward_zero_coupon(
                        self.matrix.loc[i, 'forward'], self.matrix.loc[i, 'date_diff'],
                        self.matrix.loc[(i-1), 'zero'])

                    self.matrix.loc[i, 'yield'] = mult*(-1)*np.log(self.matrix.loc[i, 'zero'])/\
                        self.matrix.loc[i, 'maturity']
        else:
            raise ValueError("Vector type not supported")

        self.matrix.index = self.schedule

        if origin is not None and isinstance(origin, list):
            for indx in self.matrix.index:
                if indx in origin:
                    self.matrix.loc[indx, 'origin'] = 1
                else:
                    self.matrix.loc[indx, 'origin'] = 2

    def status(self):
        ''' determines health of matrix '''
        res = (all(np.logical_not(np.isnan(self.matrix))) and\
                np.logical_and(all(self.matrix.zero > -0.00005), all(self.matrix.zero < 1.001)))

        return res

    def determine_closest_maturity(self, maturity):
        ''' for provided maturity determines closest maturity '''
        res = None
        for indx, val in self.matrix.iterrows():
            if abs(maturity - val['maturity']) < 0.00001:
                res = indx
                break
        return res

    def calc_zero(self, maturity):
        ''' calculates zero coupon bond based on constructed matrix'''
