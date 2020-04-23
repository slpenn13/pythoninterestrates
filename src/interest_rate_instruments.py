''' Common pricing methods corresponding to Interest rate Instruments '''
# import datetime as dt
# import businessdate as bdte
import numpy as np
import pandas as pd
import interest_rate_utilities as intutil


class fi_instrument():
    ''' initial fixed income / interest rate instrument class '''
    instrument_type = intutil.rate_instruments.ZERO_COUPON

    def __init__(self, name, first, maturity, options, princ=1.0, frequency='Y',
                 columns=None, dbg=False):
        ''' Constructor for fixed income (FI) instrument
        name: text name of instance
        first: first FI event
        maturity: maturity of FI instrument
        options
        princ: Notional of instrument, defualts to 1.0
        columns: defaults to list ['maturity', 'time_diff', 'CF'], provided set muyst be
            superset
        dbg: indicates whether to output debug output
        '''

        if columns is not None and isinstance(columns, list):
            self.columns = columns.copy()
        else:
            if dbg:
                print("Warning -- using default headers")
            self.columns = ['maturity', 'time_diff', 'CF']

        self.debug = dbg
        self.name = name
        self.princ = princ
        self.frequency = frequency

        if 'control' not in options.keys():
            raise ValueError("options dictionary must include key defaults")
        self.options = options.copy()
        if 'convention' not in options['control'].keys():
            self.options['control']['convetion'] = '30360'

        self.schedule = self.generate_schedule(first, maturity)
        self.cash_flow_df = np.zeros([len(self.schedule), len(self.columns)])
        self.build_cf_matrix()


    def generate_schedule(self, start, maturity, princ=1.):
        ''' generates schedule of events (payments)
        start: date of first event
        maturity: date of final event, e.g. final payment
        '''
        count = intutil.calc_bdte_diff_int(maturity, start, self.options, princ, dbg=self.debug)
        if self.frequency.upper().startswith('M'):
            per = 12*count
        elif self.frequency.upper().startswith('Q'):
            per = 4*count
        elif self.frequency.upper().startswith('S'):
            per = 12*count
        elif self.frequency.upper().startswith('W'):
            per = 52*count
        elif self.frequency.upper().startswith('D'):
            per = 365*count
        else:
            per = count

        return intutil.calc_schedule(start, per, self.options, self.frequency)

    def build_cf_matrix(self):
        ''' build CF df w/ dates '''

        dates = []
        matrix = self.cash_flow_df.copy()
        now = intutil.convert_date_bdte(self.options['start_date'], self.options)
        prev = now

        for loc, itm in zip(np.arange(0, len(self.schedule)), self.schedule):
            matrix[loc][0] = now.get_day_count(
                itm, self.options['control']['convention'])
            matrix[loc][1] = prev.get_day_count(
                itm, self.options['control']['convention'])

            prev = itm
            dates.append(itm.to_date())

        self.cash_flow_df = pd.DataFrame(matrix, index=dates, columns=self.columns)

    def generate_cf(self):
        ''' returns principal as final cashflow date '''
        max_date = np.max(self.schedule)
        self.cash_flow_df.loc[max_date, 'CF'] = self.princ

    def price_zeros(self, zero):
        ''' prices CF assuming 1. zero coupon bond 2. zero cp bond has SAME maturity as CF'''
        max_date = np.max(self.schedule)
        return self.cash_flow_df.loc[max_date, 'CF']*zero


class fixed_coupon_bond(fi_instrument):
    ''' class corresponding to fixed rate coupon bond on princ, with final payment
        (1 + 0.01*coupon)*princ
    '''
    instrument_type = intutil.rate_instruments.FIXED_RATE_BOND

    def __init__(self, name, first, maturity, options, coupon, princ=1.0, dbg=False):
        ''' fixed coupon bond constructor
        name: str reference name used in identification upstream
        first: first cashflow date
        maturity: final cash flow date at which principal is released
        options: options control dictionary
        coupon: item of type fixed_coupon or double
        princ: notional on which coupon payments are based
        dbg: controls debugging of instrument
        '''
        if isinstance(coupon, intutil.fixed_coupon):
            self.coupon = intutil.fixed_coupon(coupon.coupon, coupon.frequency,
                                               coupon.convention, coupon.adjust,
                                               coupon.in_percent)
        elif isinstance(coupon, float):
            self.coupon = intutil.fixed_coupon(coupon=coupon)

        super().__init__(name, first, maturity, options, princ=princ,
                         frequency=self.coupon.frequency, dbg=dbg)
        self.generate_cf()

    def generate_cf(self):
        max_date = np.max(self.schedule)
        for row in self.cash_flow_df.iterrows():
            if self.schedule[0] < row[0]:
                if max_date > row[0]:
                    self.cash_flow_df.loc[row[0], 'CF'] =\
                            self.coupon.calc_coupon(row[1][1])*self.princ
                else:
                    self.cash_flow_df.loc[row[0], 'CF'] = self.princ +\
                            self.coupon.calc_coupon(row[1][1])*self.princ
            if self.debug:
                print(row[1])

class floating_rate_bond(fi_instrument):
    '''class corresponds to floating rate coupon bond, with princ returned at final payment'''
    instrument_type = intutil.rate_instruments.FLOATING_RATE_BOND

    def __init__(self, name, first, maturity, options, coupon, princ=1.0, dbg=False):
        ''' floating rate bond constructor -- differs from FIXED RATE_BOND in coupon_dict '''
        if isinstance(coupon, intutil.floating_coupon):
            self.coupon = intutil.floating_coupon(
                coupon.reference_rate, coupon.a, coupon.b, coupon.frequency,
                coupon.convention, coupon.adjust, coupon.in_percent)
        elif isinstance(coupon, str) and coupon in ['LIBOR_1MO', 'LIBOR_3MO', 'LIBOR_MO']:
            self.coupon = intutil.floating_coupon(coupon)
        else:
            raise ValueError("Coupon Dict must be of type floating_coupon")

        super().__init__(name, first, maturity, options, princ=princ,
                         frequency=self.coupon.frequency,
                         columns=['maturity', 'time_diff', 'CF', 'rate', 'coupon'], dbg=dbg)

    def calc_coupon(self):
        ''' calculates coupon based on interest rate data based on formula a*reference_rate+b'''


class swap(fi_instrument):
    ''' Simple swpa instruemnt that accepts legs as inputs'''
    instrument_type = intutil.rate_instruments.SWAP

    def __init__(self, name, leg1, leg2, options, dbg=False):
        ''' Swap constructor '''
        self.legs = []

        if isinstance(leg1, fi_instrument) and isinstance(leg2, fi_instrument):
            if dbg:
                print(name, leg1.name, leg2.name)
        else:
            raise ValueError("leg1 && leg2 must be inherited from fi_instruemnt")

        self.legs.append(leg1)
        max_lg1 = max(leg1.schedule)
        min_lg1 = min(leg1.schedule)
        max_lg2 = max(leg2.schedule)
        min_lg2 = min(leg2.schedule)

        self.legs.append(leg2)
        if len(self.legs) == 2 and\
                self.legs[0].coupon.frequency == self.legs[1].coupon.frequency and\
                self.legs[0].coupon.convention == self.legs[1].coupon.convention and\
                max_lg1 == max_lg2 and\
                min_lg1 == min_lg2:
            if self.legs[0].instrument_type == intutil.rate_instruments.FIXED_RATE_BOND:
                self.r_swap = self.legs[0].coupon.coupon
                self.notional = self.legs[0].princ
                self.fixed_loc = 0
                self.float_loc = 1
            elif self.legs[1].instrument_type == intutil.rate_instruments.FIXED_RATE_BOND:
                self.r_swap = self.legs[1].coupon.coupon
                self.notional = self.legs[1].princ
                self.fixed_loc = 1
                self.float_loc = 0
            else:
                raise ValueError("One of instruments must be FIXED or both floating")


            if 'is_fixed_payer' in options.keys():
                self.is_fixed_payer = bool(int(options['is_fixed_payer']) > 0)
            else:
                self.is_fixed_payer = True

            self.reset = min_lg1
            self.maturity = max(max_lg1, max_lg2)
            super().__init__(name, min_lg1, max_lg1, options, self.notional,
                             self.legs[0].coupon.frequency,
                             columns=['maturity', 'time_diff', 'CF', 'CF_fixed', 'CF_floating'],
                             dbg=dbg)
        else:
            raise ValueError("Coupon payment dates are not congruent")

        self.update_cash_df()

    def update_cash_df(self):
        ''' Updates cash flow matrix generated with inputs from each leg '''
        mult = (1.0 if self.is_fixed_payer else -1.0)

        if self.cash_flow_df is not None and isinstance(self.cash_flow_df, pd.DataFrame) and\
                all(self.cash_flow_df.shape) > 0:
            for row in  self.legs[self.fixed_loc].cash_flow_df.iterrows():
                if self.debug:
                    print(row)
                self.cash_flow_df.loc[row[0], 'CF_fixed'] = row[1]['CF']
                self.cash_flow_df.loc[row[0], 'CF'] = self.cash_flow_df.loc[row[0], 'CF_fixed'] -\
                    self.cash_flow_df.loc[row[0], 'CF_floating']

            for row in  self.legs[self.float_loc].cash_flow_df.iterrows():
                if row[0] == self.reset and abs(row[1]['CF']) < 0.000001:
                    self.cash_flow_df.loc[row[0], 'CF_floating'] = self.notional
                else:
                    self.cash_flow_df.loc[row[0], 'CF_floating'] = row[1]['CF']

                self.cash_flow_df.loc[row[0], 'CF'] = mult*(
                    self.cash_flow_df.loc[row[0], 'CF_fixed'] -\
                    self.cash_flow_df.loc[row[0], 'CF_floating'])


        else:
            raise ValueError("Cash flow matrix is not instantiated!!")


def build_swap(name, swap_dict, options, dbg=False):
    ''' Constructs SWAP from dictionary '''
    if swap_dict["type"].upper() == 'SWAP':
        princ = (swap_dict['princ'] if 'princ' in swap_dict.keys() else 1.0)
        cpn_fixed = intutil.fixed_coupon(swap_dict['rate'], swap_dict['frequency'])
        lg1 = fixed_coupon_bond('FIXED', swap_dict['reset_date'], swap_dict['date'],
                                options, cpn_fixed, princ, dbg=dbg)

        cpn_float = intutil.floating_coupon(swap_dict["reference_rate"], swap_dict['frequency'])
        lg2 = floating_rate_bond("FLOATING", swap_dict['reset_date'], swap_dict['date'],
                                 options, cpn_float, princ, dbg)

        swap_final = swap(name, lg1, lg2, options, dbg=dbg)
    else:
        raise ValueError("Dicxt type muyst be swap")

    return swap_final

def apply_yield_forward_calcs(df, options):
    ''' Calculates spot, yield and forwards based on provides zeros
    '''
    df = intutil.calc_yield_continuous(df, options)
    df = intutil.calc_spot_simple(df, options)
    df = intutil.calc_forward_rate(df, options)

    return df
