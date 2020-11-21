''' Common pricing methods corresponding to Interest rate Instruments '''
# import datetime as dt
# import businessdate as bdte
import numpy as np
import pandas as pd
import scipy.optimize as sco
import interest_rate_base as intbase
import interest_rate_dates as intdate
import interest_rate_hjm as inthjm
import interest_rate_capfloor_convenience as intconv
import interest_rate_discount as intdisc
import interest_rate_discount_lorimier as intdisc_lor


class fi_instrument():
    ''' initial fixed income / interest rate instrument class '''
    instrument_type = intbase.rate_instruments.ZERO_COUPON

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
            if 'discount' not in self.columns:
                self.columns.append('discount')
            if 'CF' not in self.columns:
                self.columns.append('CF')
        else:
            if dbg:
                print("Warning -- using default headers")
            self.columns = ['maturity', 'time_diff', 'discount', 'CF']

        self.debug = dbg
        self.name = name
        self.maturity_ = None
        self.yield_ = None
        self.cash_flow_df = None
        self.princ = princ
        self.price = None
        self.frequency = frequency

        if 'control' not in options.keys():
            raise ValueError("options dictionary must include key defaults")
        self.options = options.copy()
        if 'convention' not in options['control'].keys():
            self.options['control']['convetion'] = '30360'
        self.schedule = self.generate_schedule(first, maturity)
        self.maturity = max(self.schedule)
        self.build_cf_matrix()


    def generate_schedule(self, start, maturity, princ=1.):
        ''' generates schedule of events (payments)
        start: date of first event
        maturity: date of final event, e.g. final payment
        '''
        count = intdate.calc_bdte_diff_int(maturity, start, self.options, princ, dbg=self.debug)
        # print(self.name, count)

        mat = intdate.convert_date_bdte(maturity, self.options)

        if self.frequency:
            per = intdate.convert_period_count(count, self.frequency)

            if self.debug:
                print(start, per, self.frequency)

            sched = intdate.calc_schedule(start, mat, self.options, self.frequency)
        else:
            sched = [intdate.convert_date_bdte(start, self.options), mat]

        mat = intdate.adjust_date_bd_convention(mat, self.options, False)
        sched = [itm for itm in sched if itm <= mat]
        return sorted(sched)

    def build_cf_matrix(self):
        ''' build CF df w/ dates '''

        dates = []
        if self.cash_flow_df is not None and isinstance(self.cash_flow_df, pd.DataFrame) and\
                all(self.cash_flow_df.shape) > 0:
            matrix = self.cash_flow_df.copy()
        else:
            matrix = np.zeros([len(self.schedule), len(self.columns)])

        now = intdate.convert_date_bdte(self.options['start_date'], self.options)
        prev = now

        for loc, itm in enumerate(self.schedule):
            matrix[loc][0] = now.get_day_count(
                itm, self.options['control']['convention'])
            matrix[loc][1] = prev.get_day_count(
                itm, self.options['control']['convention'])

            prev = itm
            dates.append(itm.to_date())

        self.cash_flow_df = pd.DataFrame(matrix, index=dates, columns=self.columns)

    def generate_cf(self, price=None):
        ''' returns principal as final cashflow date '''
        max_date = np.max(self.schedule)
        self.cash_flow_df.loc[max_date, 'CF'] = self.princ

        if price:
            min_date = np.min(self.schedule)
            self.cash_flow_df.loc[min_date, 'CF'] = (self.princ/100.)*price

        self.calc_maturity()
        self.calc_yield()

    def calc_maturity(self):
        ''' Calculated continues maturity in years '''
        mn = self.schedule[0]
        mx = self.schedule[len(self.schedule)-1]
        self.maturity_ = self.cash_flow_df.loc[mx, 'maturity'] -\
                self.cash_flow_df.loc[mn, 'maturity']

    def calc_WAM(self):
        ''' calculates the Weighted Average Maturity '''
        den = self.cash_flow_df[1:]['CF'].sum()
        num = self.cash_flow_df[1:]['CF'].dot(self.cash_flow_df[1:]['maturity'])

        return num / den

    def calc_yield(self, price=None):
        ''' calculates continuous yield '''
        mx = np.max(self.schedule)
        if price and isinstance(price, float):
            price2 = price
        else:
            price2 = self.get_price()
        self.yield_ = 100.*np.log(price2/self.cash_flow_df.loc[mx, 'CF'])/(-1*self.maturity_)

    def get_yield(self, price=None):
        ''' gets yields '''
        if self.yield_ is None or not isinstance(self.yield_, float) or\
                np.isnan(self.yield_):
            self.calc_yield(price)

        return self.yield_

    def get_price(self):
        ''' Obtains original price '''
        mn = np.min(self.schedule)
        return self.cash_flow_df.loc[mn, 'CF']

    def set_price(self, price=None):
        ''' Sets price '''
        if price and isinstance(price, float):
            mn = np.min(self.schedule)
            self.price = price
            self.cash_flow_df.loc[mn, 'CF'] = -1.0*price
        else:
            if self.debug:
                print("Warning: faulty price")

    def calc_price_zeros(self, zero):
        ''' prices CF assuming 1. zero coupon bond 2. zero cp bond has SAME maturity as CF'''
        max_date = np.max(self.schedule)
        res = np.NAN

        if isinstance(zero, float):
            res = self.cash_flow_df.loc[max_date, 'CF']*zero
        elif isinstance(zero, intdisc.discount_calculator):
            res = self.cash_flow_df.loc[
                max_date, 'CF']*zero.calc_zero(self.cash_flow_df.loc[max_date, 'maturity'])
        else:
            raise ValueError("Faulty zeros type, must be float of disc_calculator")

        return res

    def calc_price_yields(self, yield_, include_first=False):
        ''' Calculates price give constant yield '''

        if include_first:
            zeros = np.exp(-0.01*yield_*self.cash_flow_df.maturity)
            cfs = self.cash_flow_df.CF
        else:
            zeros = np.exp(-0.01*yield_*self.cash_flow_df[1:].maturity)
            cfs = self.cash_flow_df[1:].CF

        return zeros.dot(cfs)

    def get_maturity(self):
        ''' Calculated continues maturity in years '''
        if self.maturity_ is None or not isinstance(self.maturity_, float) or\
                np.isnan(self.maturity_):
            self.calc_maturity()

        return self.maturity_
   #     determine_closest_maturity:


class fixed_coupon_bond(fi_instrument):
    ''' class corresponding to fixed rate coupon bond on princ, with final payment
        (1 + 0.01*coupon)*princ
    '''
    instrument_type = intbase.rate_instruments.FIXED_RATE_BOND

    def __init__(self, name, first, maturity, options, coupon, princ=1.0,
                 price=np.NAN, dated=None, dbg=False):
        ''' fixed coupon bond constructor
        name: str reference name used in identification upstream
        first: first cashflow date
        maturity: final cash flow date at which principal is released
        options: options control dictionary
        coupon: item of type fixed_coupon or double
        princ: notional on which coupon payments are based
        dbg: controls debugging of instrument
        '''
        if isinstance(coupon, intbase.fixed_coupon):
            self.coupon = intbase.fixed_coupon(coupon.coupon, coupon.frequency,
                                               coupon.convention, coupon.adjust,
                                               coupon.in_percent)
        elif isinstance(coupon, float):
            self.coupon = intbase.fixed_coupon(coupon=coupon)
        else:
            raise ValueError("Faulty Coupon")

        super().__init__(name, first, maturity, options, princ=princ,
                         frequency=self.coupon.frequency, dbg=dbg)

        if dated is None:
            self.dated = intdate.convert_date_bdte(first, self.options)
        else:
            self.dated = intdate.convert_date_bdte(dated, self.options)

        if price and not np.isnan(price):
            self.price = price
        else:
            self.price = None

        self.generate_cf(self.price)

    def generate_cf(self, price=None):
        if price:
            min_date = np.min(self.schedule)
            self.cash_flow_df.loc[min_date, 'CF'] = -1.*(self.princ/100.)*price

        for row in self.cash_flow_df.iterrows():
            if self.dated < row[0]:
                if self.maturity > row[0]:
                    self.cash_flow_df.loc[row[0], 'CF'] =\
                            self.coupon.calc_coupon(row[1][1])*self.princ
                else:
                    self.cash_flow_df.loc[row[0], 'CF'] = self.princ +\
                            self.coupon.calc_coupon(row[1][1])*self.princ
            if self.debug:
                print(row[1])

    def calc_accrued_interest(self, accrual_start=None, settle=None, coupon_date=None):
        ''' calculates accrued interest'''

        if accrual_start and isinstance(accrual_start, float) and settle and\
                isinstance(settle, float):
            mult = (settle - accrual_start)
        elif not accrual_start and coupon_date and\
                isinstance(coupon_date, (intdate.dt.date, intdate.bdte.BusinessDate)):
            cpn_date = (intdate.bdte.BusinessDate(coupon_date)
                        if isinstance(coupon_date, intdate.dt.date) else coupon_date)

            if self.coupon.per > 0.95:
                prd = intdate.bdte.BusinessPeriod(years=1.0)
            elif self.coupon.per > 0.075:
                prd = intdate.bdte.BusinessPeriod(months=int(self.coupon.per*12))
            elif self.coupon.per < 0.0027:  # Days
                if self.options['control']['convention'].lower().endswith("act") or\
                        self.options['control']['convention'].lower().endswith("365"):
                    val = 365
                else:
                    val = 360
                prd = intdate.bdte.BusinessPeriod(months=int(self.coupon.per*val))
            else:
                raise ValueError("Faulty Coupon Period")
            accrual_start = cpn_date - prd
            accrual_start = accrual_start.adjust()
            if settle:
                settle = intdate.convert_date_bdte(settle, self.options)
            else:
                settle = intdate.convert_date_bdte(self.options['start_date'], self.options)

            mult = accrual_start.get_day_count(settle, self.options['control']['convention'])
        else:
            if self.debug:
                print(type(accrual_start), type(settle), type(coupon_date))
            raise ValueError("Faulty accrued combination")

        return self.princ*self.coupon.calc_coupon(mult)

    def calc_yield(self, price=None):
        ''' calculates continuous yield '''
        if price and isinstance(price, float):
            price2 = self.get_price()
            self.set_price(price)

        x1 = sco.brentq(self.calc_price_yields, -15.0, 100.0, xtol=0.0000001, args=(True))
        if x1 and isinstance(x1, float) and not np.isnan(x1):
            self.yield_ = x1
            self.price = price

        self.set_price(price2)

        return x1


    def calc_price_zeros(self, zero):
        ''' prices CF assuming '''
        if zero and isinstance(zero, list) and len(zero) == (self.cash_flow_df.shape[0]-1):
            for i, val in enumerate(zero):
                self.cash_flow_df.loc[self.schedule[i+1], 'discount'] = val
        elif zero and isinstance(zero, np.ndarray) and (zero.size == self.cash_flow_df.shape[0]-1):
            for i, val in enumerate(zero):
                self.cash_flow_df.loc[self.schedule[i+1], 'discount'] = val

        elif isinstance(zero, (intdisc.discount_calculator,
                               intdisc_lor.discount_calculator_lorimier)):

            # index = [intdate.bdte.BusinessDate(i) for i in self.cash_flow_df.index]
            # zero_vctr = zero.calc_vector_zeros(self.cash_flow_df[1:].index)
            zero_vctr = zero.calc_vector_zeros(self.cash_flow_df[1:]['maturity'])

            for i, val in enumerate(zero_vctr):
                self.cash_flow_df.loc[self.schedule[i+1], 'discount'] = val

        else:
            print(type(zero))
            raise ValueError("Faulty Discounting Method (Zeros)")

        if self.debug:
            print(self.cash_flow_df.describe())
            return self.cash_flow_df['CF'].dot(self.cash_flow_df['discount'])

        return zero_vctr.dot(self.cash_flow_df[1:]['CF'])

class floating_rate_bond(fi_instrument):
    '''class corresponds to floating rate coupon bond, with princ returned at final payment'''
    instrument_type = intbase.rate_instruments.FLOATING_RATE_BOND

    def __init__(self, name, first, maturity, options, coupon, princ=1.0, dbg=False):
        ''' floating rate bond constructor -- differs from FIXED RATE_BOND in coupon_dict '''
        if isinstance(coupon, intbase.floating_coupon):
            self.coupon = intbase.floating_coupon(
                coupon.reference_rate, coupon.a, coupon.b, coupon.frequency,
                coupon.convention, coupon.adjust, coupon.in_percent)
        elif isinstance(coupon, str) and coupon in ['LIBOR_1MO', 'LIBOR_3MO', 'LIBOR_MO']:
            self.coupon = intbase.floating_coupon(coupon)
        else:
            raise ValueError("Coupon Dict must be of type floating_coupon")

        super().__init__(name, first, maturity, options, princ=princ,
                         frequency=self.coupon.frequency,
                         columns=['maturity', 'time_diff', 'CF', 'rate', 'coupon'], dbg=dbg)

    def calc_coupon(self):
        ''' calculates coupon based on interest rate data based on formula a*reference_rate+b'''


class swap(fi_instrument):
    ''' Simple swpa instruemnt that accepts legs as inputs'''
    instrument_type = intbase.rate_instruments.SWAP

    def __init__(self, name, leg1, leg2, options, is_market=True, t0_equal_T0=None,
                 reset=None, dbg=False):
        ''' Swap constructor
        name: str determinig name of SWAP
        leg1: first leg in the swap
        leg2: second leg in the swap CF (leg1 - leg2)
        '''
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
            if self.legs[0].instrument_type == intbase.rate_instruments.FIXED_RATE_BOND:
                self.r_swap = self.legs[0].coupon.coupon
                self.notional = self.legs[0].princ
                self.fixed_loc = 0
                self.float_loc = 1
            elif self.legs[1].instrument_type == intbase.rate_instruments.FIXED_RATE_BOND:
                self.r_swap = self.legs[1].coupon.coupon
                self.notional = self.legs[1].princ
                self.fixed_loc = 1
                self.float_loc = 0
            else:
                raise ValueError("One of instruments must be FIXED or both floating")

            if reset is None:
                self.reset = min_lg1
            else:
                self.reset = intdate.convert_date_bdte(reset, options)

            self.maturity = max(max_lg1, max_lg2)

            super().__init__(name, self.reset, self.maturity, options, self.notional,
                             self.legs[0].coupon.frequency,
                             columns=['maturity', 'time_diff', 'CF', 'CF_fixed', 'CF_floating',
                                      'price'],
                             dbg=dbg)

            if 'is_fixed_payer' in self.options.keys():
                self.is_fixed_payer = bool(int(self.options['is_fixed_payer']) > 0)
            else:
                self.is_fixed_payer = True

            if t0_equal_T0 is None:
                if 'start_date' in self.options.keys():
                    dte = intdate.convert_date_bdte(self.options['start_date'], self.options)
                    self.t0_equal_T0 = bool(dte == self.reset)
                else:
                    self.t0_equal_T0 = False
            else:
                self.t0_equal_T0 = t0_equal_T0

            self.is_market_quote = (intbase.load_types.MARKET if\
                                    is_market else
                                    intbase.load_types.INTERPOLATED)

            self.reset = intdate.adjust_date_bd_convention(self.reset, self.options, self.debug)

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
                    if self.t0_equal_T0:
                        self.cash_flow_df.loc[row[0], 'price'] = self.notional
                    else:
                        self.cash_flow_df.loc[row[0], 'CF_floating'] = self.notional
                else:
                    self.cash_flow_df.loc[row[0], 'CF_floating'] = row[1]['CF']

                self.cash_flow_df.loc[row[0], 'CF'] = mult*(
                    self.cash_flow_df.loc[row[0], 'CF_fixed'] -\
                    self.cash_flow_df.loc[row[0], 'CF_floating'])


        else:
            raise ValueError("Cash flow matrix is not instantiated!!")

    def calc_swap_strike_zeros(self, zeros=None, update=False):
        ''' calculatules ATM strike for swap
        zeros: df including zolumn zero used to discount cash flows
        '''
        numer = (zeros.loc[self.reset, 'zero'] - zeros.loc[self.maturity, 'zero'])
        # print(type(zeros.index[0]), type(self.reset), type(self.maturity))
        ind = np.logical_and(zeros.index > self.reset,
                             zeros.index <= self.maturity)

        denom = zeros.loc[ind, 'zero'].dot(zeros.loc[ind, 'date_diff'])
        if self.debug:
            print(zeros.loc[ind, 'zero'])
            print(zeros.loc[ind, 'date_diff'])
            print("Strike (zeros): numer %f denom %f (len(ind %d)) " % (
                numer, denom, ind.sum()))

        res = numer / denom
        if update and np.isfinite(res):
            if self.debug:
                print("Warning -- updating r_swap and cash flows")
            self.r_swap = res
            self.legs[self.fixed_loc].coupon.coupon = res
            self.legs[self.fixed_loc].generate_cf()
            self.update_cash_df()


        return res

    def calc_swap_strike_forwards(self, zeros=None, update=False):
        ''' calculates swap strike as weighted average of forward curve '''
        ind = np.logical_and(zeros.matrix.index > self.reset,
                             zeros.matrix.index <= self.maturity)
        res = np.average(zeros.matrix.loc[ind, 'forward'], weights=zeros.matrix.loc[ind, 'zero'])

        if update and np.isfinite(res):
            if self.debug:
                print("Warning -- updating r_swap and cash flows")
            self.r_swap = res
            self.legs[self.fixed_loc].coupon.coupon = res
            self.legs[self.fixed_loc].generate_cf()
            self.update_cash_df()

        return res

class caplet(fi_instrument):
    ''' caplet class '''
    instrument_type = intbase.rate_instruments.CAPLET

    def __init__(self, name, strike, maturity, reset, options, is_floor=False, princ=1.0,
                 frequency='Y', dbg=False):
        ''' Caplet constructor
        name: name of capletshort_rate_model as  short_rate
        strike: strike applied in pay-off lambda function
        maturity: maturity of caplet
        reset: reset of caplet
        options: control dctionary (must include control in keys())
        is_floor: determines whether caplet is caplet cap or caplet floor (determines pay-off)
        princ: principal applied to pay-off
        frequency: period between reset and maturity
        dbg: determines whether to output debugf information
        '''
        if options is not None and isinstance(options, dict) and 'control' in options.keys():
            self.reset = reset
            self.maturity = intdate.adjust_date_bd_convention(maturity, options, dbg)
            super().__init__(name, self.reset, self.maturity, options, princ=princ,
                             frequency=frequency,
                             columns=['maturity', 'time_diff', 'CF', 'CF_fixed', 'CF_floating'],
                             dbg=dbg)

            self.strike = strike
            if is_floor is None:
                self.is_floor = False
                self.calc_CF = lambda x: (x if x <= self.strike else self.strike)
            else:
                self.is_floor = is_floor
                if self.is_floor:
                    self.calc_CF = lambda x: (x if x >= self.strike else self.strike)
                else:
                    self.calc_CF = lambda x: (x if x <= self.strike else self.strike)
        else:
            raise ValueError("Options fails criteria for caplet construction")

        self.generate_cf()

    def generate_cf(self):
        ''' generates initial cap let cash flow '''
        mult = (-1.0 if self.is_floor else 1.0)
        self.cash_flow_df.loc[self.maturity.to_date(), 'CF_fixed'] = mult*self.princ*self.strike

    def price_caplet(self, t1, t2, zero_t1, zero_t2, mdl=None, kappa=None, sigma=None):
        ''' pricer -- accepts model override (mdl)'''
        if isinstance(mdl, inthjm.hjm_model):
            price = mdl.calc_price_caplet(self.strike, t1, t2,
                                          zero_t1, zero_t2,
                                          kappa=kappa, sigma=sigma)
        else:
            if isinstance(mdl, str) and mdl.lower().startswith('bache'):
                price = intconv.calc_price_bachelier_caplet(self.strike, t1, t2, zero_t1,
                                                            zero_t2, sigma, dbg=self.debug)

            else:
                if self.debug:
                    print("Warning using simple caplet pricing")
                    if kappa is not None:
                        print("Warning kappa not used in simple caplet calculation")
                price = intconv.calc_price_black_caplet(self.strike, t1, t2, zero_t1,
                                                        zero_t2, sigma, dbg=self.debug)

        return price

    def vega_caplet(self, t1, t2, zero_t1, zero_t2, mdl=None, sigma=None, dbg=False):
        ''' calculate vega for caplet '''

        dbg = (self.debug or dbg)
        if isinstance(mdl, inthjm.hjm_model):
            if self.debug:
                print("Warning -- vega calculation implemented HJM ")
        else:
            if isinstance(mdl, str) and mdl.upper().startswith("BACHE"):
                vega = intconv.calc_vega_bachelier_caplet(self.strike, t1, t2, zero_t1,
                                                          zero_t2, sigma, dbg=dbg)
            else:
                vega = intconv.calc_vega_black_caplet(self.strike, t1, t2, zero_t1, zero_t2,
                                                      sigma, dbg=dbg)

        return vega

class cap(fi_instrument):
    ''' cap structure -- based on dictionary of caplets '''
    instrument_type = intbase.rate_instruments.CAP

    def __init__(self, name, strike, maturity, reset, options, princ=1.0, frequency='Y',
                 dbg=False):
        ''' cap constructor '''

        self.reset = reset
        self.maturity = intdate.adjust_date_bd_convention(maturity, options, dbg)
        self.caplet = {}
        self.sched = None
        super().__init__(name, self.reset, self.maturity, options, princ=princ,
                         frequency=frequency,
                         columns=['maturity', 'time_diff', 'CF', 'CF_fixed', 'CF_floating'],
                         dbg=dbg)

        self.strike = 0.01*strike
        self.apply_schedule()


    def calc_schedule(self):
        ''' calculates unadjsuted BusinessDate schedule'''
        if self.sched is None:
            periods = intdate.calc_bdte_diff_int(self.maturity, self.reset, self.options,
                                                 dbg=self.debug)

            per = intdate.convert_period_count(periods, self.frequency)

            reset = intdate.convert_date_bdte(self.reset, self.options)
            self.sched = intdate.calc_schedule(reset, per, options=None,
                                               period=self.frequency)
            self.sched = [itm for itm in self.sched if itm <= self.maturity]
            if self.debug:
                print(per, periods, (0 if self.sched is None else len(self.sched)))

    def apply_schedule(self):
        ''' calculates schedule of caplets and stores in caplets dictionary '''
        if self.sched is None:
            self.calc_schedule()

        reset = self.reset

        for loc, dte in zip(np.arange(0, len(self.sched)), self.sched):
            name = "".join(["CAPLET", str(loc)])
            if loc > 0:
                self.caplet[name] = caplet(name, self.strike, dte, reset, self.options,
                                           is_floor=False, princ=self.princ,
                                           frequency=self.frequency, dbg=False)

            reset = dte

    def price_cap(self, zeros, sigma=None, kappa=None, hjm_model=None):
        ''' calculates cap price as sum of caplet prices
        hjm_model: type(hjm_model) == 'hjm_model'
        zeros: DataFrame with one column zero coupon bond prices
        sigma: can be left None, if set overrides model parameter
        kappa: can be left None, if set overrides model parameter
        '''
        result = 0.0

        for itm in self.caplet.values():
            zero_t1 = zeros.loc[itm.cash_flow_df.index[0], 'zero']

            result += itm.price_caplet(itm.cash_flow_df.iloc[0, 0],
                                       itm.cash_flow_df.iloc[1, 0], zero_t1,
                                       zeros.loc[itm.cash_flow_df.index[1], 'zero'],
                                       hjm_model, sigma=sigma, kappa=kappa)

        return result

    def price_cap_solver(self, sigma, zeros, price=0.0, kappa=None, hjm_model=None, dbg=False):
        ''' cap price calculator '''
        dbg = (dbg or self.debug)
        result = self.price_cap(zeros, sigma=sigma, kappa=kappa, hjm_model=hjm_model)
        if dbg:
            print("sigma %.8f target %f Value %.8f Diff %.8f" % (
                sigma, price, result, (price - result)))

        return price - result

    def calc_implied_volatility(self, zeros, price=0.0, left=0.0005, right=2.0, tol=1.e-5,
                                hjm_model=None, dbg=False):
        ''' calculates implied volatility for given price '''
        xresult = sco.brentq(self.price_cap_solver, left, right, args=(
            zeros, price, None, hjm_model, dbg), full_output=True)

        if dbg:
            print(xresult)

        return xresult[0]

    def vega_cap(self, zeros, sigma=None, kappa=None, hjm_model=None, dbg=False):
        ''' calculates vega for cap as sum of caplet vegas '''
        result = 0.0

        dbg = (self.debug or dbg)
        for itm in self.caplet.values():
            result += itm.vega_caplet(itm.cash_flow_df.iloc[0, 0],
                                      itm.cash_flow_df.iloc[1, 0],
                                      zeros.loc[itm.cash_flow_df.index[0], 'zero'],
                                      zeros.loc[itm.cash_flow_df.index[1], 'zero'],
                                      hjm_model, sigma=sigma, dbg=dbg)

        return result


class interest_rate_future(fi_instrument):
    ''' interest futures calculation '''
    instrument_type = intbase.rate_instruments.FUTURE

    def __init__(self, name, futures_rate, maturity, reset, frequency, options, dbg=False):
        ''' constructor '''
        self.futures_rate = futures_rate
        self.rate = intbase.futures_rate(futures_rate)
        self.options = options.copy()
        self.reset = intdate.convert_date_bdte(reset, self.options)
        self.maturity = intdate.convert_date_bdte(maturity, self.options)

        super().__init__(name, self.reset, self.maturity, self.options, princ=1.0,
                         frequency=frequency, columns=['maturity', 'time_diff', 'CF', 'price'],
                         dbg=dbg)

        self.spot = self.calc_futures_spot()
        self.generate_cf()

    def __repr__(self):
        mat = "-".join([str(self.maturity.year), str(self.maturity.month), str(self.maturity.day)])
        reset = "-".join([str(self.reset.year), str(self.reset.month), str(self.reset.day)])

        res = " ".join([self.name, "Maturity", mat, "Reset", reset, "Futures",
                        str(round(self.futures_rate, 4)), "Spot", str(round(self.spot, 4))])

        return res

    def generate_cf(self):
        ''' generates initial futures cash flow '''

        self.cash_flow_df.loc[self.reset.to_date(), 'CF'] = self.princ*-1.0
        self.calc_futures_payoff()

    def calc_futures_spot(self):
        ''' calculate spot rate '''
        spot = np.NAN
        proj_date = intdate.calc_schedule(self.reset, 1, self.options, period=self.frequency)
        proj_date_fnl = max(proj_date)

        if abs(proj_date_fnl - self.maturity) > intdate.bdte.BusinessPeriod(days=2):
            if self.debug:
                print("Futures -- date mismatch: maturity %s projeted %s" % (
                    self.maturity.to_date(), proj_date_fnl.to_date()))

            diff = self.reset.get_day_count(proj_date_fnl, self.options['control']['convention'])
        else:
            diff = self.reset.get_day_count(self.maturity, self.options['control']['convention'])

        spot = (1./diff)*self.rate

        return spot

    def calc_futures_payoff(self, maturity=None):
        ''' calculate futures payoff given updated maturity as year fraction '''

        if maturity and isinstance(maturity, (str, intdate.bdte.BusinessDate, intdate.dt.date)):
            mat = intdate.convert_date_bdte(maturity, self.options)
        else:
            mat = self.maturity

        mult = 0.01*intdate.calc_bdte_diff(mat, self.options, self.reset)

        if mat != max(self.cash_flow_df.index):
            print("Warning -- mismatch %s %s  shape %d" % (
                mat.to_date(), max(self.cash_flow_df.index), self.cash_flow_df.shape[0]))
            if self.cash_flow_df.shape[0] == 1:
                self.cash_flow_df.loc[mat.to_date(), 'maturity'] =\
                    intdate.calc_bdte_diff(mat, self.options)
            elif self.cash_flow_df.shape[0] == 2:
                self.cash_flow_df.index = [self.reset.to_date(), mat.to_date()]
                self.cash_flow_df.loc[mat.to_date(), 'maturity'] =\
                    intdate.calc_bdte_diff(mat, self.options)

            else:
                raise ValueError("Faulty dimansions for futures contract")

            self.cash_flow_df.loc[mat.to_date(), 'time_diff'] =\
                self.cash_flow_df.loc[mat.to_date(), 'maturity'] -\
                    self.cash_flow_df.loc[self.reset.to_date(), 'maturity']

        self.cash_flow_df.loc[mat.to_date(), 'CF'] = self.princ +\
            mult*self.princ*self.spot
