''' Common pricing methods corresponding to Interest rate Instruments '''
from enum import Enum, unique
import datetime as dt
import numpy as np
import pandas as pd


@unique
class rate_instruments(Enum):
    ''' Enumeration of fixed income instruments'''
    OTHER = 0
    LIBOR = 1
    FUTURE = 2
    FORWARD = 3
    SWAP = 4
    FRA = 5
    FIXED_RATE_BOND = 6
    FLOATING_RATE_BOND = 7
    ZERO_COUPON = 8
    CAPLET = 9
    CAP = 10
    FLOOR = 11
    SWAPTION = 12

    def __eq__(self, other):
        # print(self.value, other, type(other))
        result = False
        if isinstance(other, rate_instruments):
            result = (self.value == other.value)
        elif isinstance(other, int):
            result = (self.value == other)

        return result

    def __lt__(self, other):
        # print(self.value, other, type(other))
        result = False
        if isinstance(other, rate_instruments):
            result = (self.value < other.value)
        elif isinstance(other, int):
            result = (self.value < other)
        else:
            raise NotImplementedError

        return result

@unique
class load_types(Enum):
    ''' Enumeration determining the origin of data point '''
    UNKNOWN = 0
    MARKET = 1
    INTERPOLATED = 2

    def __eq__(self, other):
        result = False
        if isinstance(other, load_types):
            result = (self.value == other.value)
        elif isinstance(other, int):
            result = (self.value == other)

        return result

    def __lt__(self, other):
        # print(self.value, other, type(other))
        result = False
        if isinstance(other, load_types):
            result = (self.value < other.value)
        elif isinstance(other, int):
            result = (self.value < other)
        else:
            raise NotImplementedError

        return result


def calc_libor_zero_coupon(libor, time):
    ''' calculates the zero coupon bond values, assuming
    LIBOR rate: (in percent)
    time:       (in years)
    return: zero coupon value
    '''
    return 1. / (1. + time*0.01*libor)

def calc_forward_zero_coupon(rate, t2_t1_diff, t1_zero):
    ''' Values
    rate: Forward Rate (in percent)
    t2_t1_diff: length of time period for forward rate
    t1_zero: zero coupon w/ maturity t1 maturity
    '''
    return t1_zero / (1. + t2_t1_diff*0.01*rate)

def calc_swap_zero_coupon(rate, df, position, ref_position=None):
    ''' Calculates zero coupon value using inverted swap pricing formula
    rate: Swap rate @ t_i
    df: numpy array of elements
    position: location of  results
    ref_position: position of next immediate prior swaps (defaults to position-1)
    '''
    if ref_position is None:
        ref = position - 1
    else:
        ref = ref_position

    denom = 1. + 0.01*rate*(df[position][0] - df[ref][0])
    ind = np.logical_and(df.transpose()[3] == 4, df.transpose()[0] < df[position][0])

    if ind.sum() >= 1:
        value = 0.0
        prev = 0.0
        first_fnd = False
        for row in df[ind]:
            if first_fnd:
                value = value + (row[0] - prev)*row[2]
            else:
                value = row[0]*row[2]
                first_fnd = True
            prev = row[0]
    else:
        raise ValueError("No swaps found!!")

    return (1. - 0.01*rate*value) / denom

def futures_rate(price):
    ''' calculates futures rate given futures price '''
    return 100.*(1. - price*0.01)

def inverse_futures_rates(rate):
    ''' calculates inverse futures rates (i.e. price) given rate '''
    return 100. - rate

def calc_yield_continuous(df):
    ''' calculates continuous yield based on previously calculated zero coupon '''
    if df is not None and isinstance(df, pd.DataFrame) and np.all(df.shape):
        df.loc[:, 'yield'] = -100.*np.log(df.loc[:, 'zero']) / df.loc[:, 'maturity']
    else:
        raise ValueError("df must be valid dataframe")

    return df

def convert_annual_yield(yield_annual, period=1, mult=0.01):
    ''' converts annual yield to semi annual yield in case of 2 '''
    if period == 2:
        yld = 200.*(np.sqrt(1 + mult*yield_annual) - 1)
    else:
        yld = yield_annual
    
    return yld

def calc_spot_simple_1d(zero, maturity, mult=1.0):
    ''' Calculates simple spot rate based on zero and maturity '''
    return mult*(1. / zero - 1.) / maturity

def calc_spot_simple(df):
    ''' calculates simple spot rate based on perviously calculated zero coupon '''
    if df is not None and isinstance(df, pd.DataFrame) and np.all(df.shape):
        df.loc[:, 'spot'] = 100.*(1. / df.loc[:, 'zero'] - 1.) / df.loc[:, 'maturity']
    else:
        raise ValueError("df must be valid dataframe")

    return df

def calc_forward_rate_1d(t1, t2, zero_t1, zero_t2, mult=1.0):
    ''' calculates forward rate based on two consecutive zeros
    t1: maturity of first zero
    t2: maturity of second zero
    zero_t1: first zero coupon bond -- maturity t1
    zero_t2: second zero coupon bond -- maturity t2
    mult: multiple of result (def: 1.) 100. produces results in percent
    '''
    return (mult/(t2 - t1))*(zero_t1/zero_t2 - 1.)

def calc_forward_rate(df, options):
    ''' calculates simple forward rates based on previously calculated zero coupon '''
    if df is not None and isinstance(df, pd.DataFrame) and np.all(df.shape):
        prev = None
        for itm in df.iterrows():
            curr = itm[1]
            if isinstance(itm[0], dt.date):
                day = (('0' + str(itm[0].day)) if itm[0].day < 10 else str(itm[0].day))
                dte = options['control']["split"].join([
                    str(itm[0].year), str(itm[0].month), day])
            else:
                dte = options['control']["split"].join(['1999', '01', '01'])

            if dte in options["forward_references"].keys():
                if options['forward_references'][dte]:
                    prev = df[dte]
                    df.at[itm[0], 'forward'] = 100./(
                        float(curr['maturity']) -
                        float(prev['maturity']))*(float(prev['zero'])/float(curr['zero']) - 1.)
                else:
                    df.at[itm[0], 'forward'] = 100./(float(curr['maturity']))*(
                        1.0 / float(curr['zero']) - 1.)
            else:
                if prev is not None and isinstance(prev, (pd.Series, pd.DataFrame)):
                    try:
                        df.at[itm[0], 'forward'] = 100./(
                            float(curr['maturity']) -
                            float(prev['maturity']))*(float(prev['zero'])/float(curr['zero']) - 1.)
                    except ZeroDivisionError:
                        # print(curr, prev)
                        df.at[itm[0], 'forward'] = np.nan
                else:
                    df.at[itm[0], 'forward'] = 100./(float(curr['maturity']))*(
                        1.0 / float(curr['zero']) - 1.)
            prev = itm[1].copy()
            # print(type(prev))
    else:
        raise ValueError("df must be valid dataframe")

    return df


class coupon():
    '''base class for coupon '''

    def __init__(self, frequency='Y', convention='follow', adjust=True):
        self.frequency = frequency
        self.convention = convention
        self.adjust = adjust

        if self.frequency.upper().startswith('M'):
            self.per = 1/12.
        elif self.frequency.upper().startswith('Q'):
            self.per = 0.25
        elif self.frequency.upper().startswith('S'):
            self.per = 0.5
        elif self.frequency.upper().startswith('W'):
            self.per = 1/52.
        elif self.frequency.upper().startswith('D'):
            self.per = 1/365.
        else:
            self.per = 1.0


    def get_frequency(self):
        ''' return coupon frequency '''
        return self.frequency

    def calc_coupon(self, adjustment=np.nan):
        ''' returns initial adjustment '''
        if self.adjust and np.logical_not(np.isnan(adjustment)):
            return adjustment

        return self.per


class fixed_coupon(coupon):
    '''base FRB coupon '''

    def __init__(self, coupon=0.0, frequency='Y', convention='follow',
                 adjust=True, in_percent=True):
        super().__init__(frequency, convention, adjust)
        self.coupon = coupon
        self.in_percent = in_percent

    def calc_coupon(self, adjustment=np.nan):
        ''' calculates fixed coupon '''
        mult = super().calc_coupon(adjustment)
        res = (0.01*self.coupon if self.in_percent else self.coupon)
        return mult*res

class floating_coupon(coupon):
    '''base floating coupon '''

    def __init__(self, reference_rate='LIBOR', a=1.0, b=0.0,
                 frequency='Y', convention='follow',
                 adjust=True, in_percent=True):
        super().__init__(frequency, convention, adjust)
        self.a = a
        self.b = b
        self.reference_rate = reference_rate
        self.in_percent = in_percent

    def calc_coupon_floating(self, adjustment=np.nan, rate=0.0):
        ''' calculates floating rate coupon based on formula ax+b '''
        mult = super().calc_coupon(adjustment)
        res = (0.01*self.a if self.in_percent else self.a)
        return mult*(res*rate + self.b)
