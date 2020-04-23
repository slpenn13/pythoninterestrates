''' Common pricing methods corresponding to Interest rate Instruments '''
from enum import Enum, unique
import datetime as dt
import businessdate as bdte
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

    def __eq__(self, other):
        # print(self.value, other, type(other))
        return self.value == other.value

    def __lt__(self, other):
        # print(self.value, other, type(other))
        return self.value < other.value
@unique
class load_types(Enum):
    ''' Enumeration determining the origin of data point '''
    UNKNOWN = 0
    MARKET = 1
    INTERPOLATED = 2


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

def convert_date_bdte(date, options):
    ''' converts date to  '''
    if isinstance(date, bdte.BusinessDate):
        ret_date = date
    elif isinstance(date, (dt.date, dt.datetime)):
        ret_date = bdte.BusinessDate(date)
    else:
        ret_date = bdte.BusinessDate(dt.datetime.strptime(
            date, options['control']['date_format']))

    return ret_date

def calc_bdte_diff(date, options, start=None, dc_convention=None):
    ''' Calculates maturity in years relative start_date included in options dictionary
    date: date in Y%-%m-%d format
    options: controls dictionary
    start: default (None) start date -- if None calcualtes maturity relative to
            options["start_date"]
    dc_convention: default (None) daycount convention aplied in diff calculation
            -- if None applies options['control'][convention"]
    returns date_diff
    '''
    if start is None:
        start = convert_date_bdte(options["start_date"], options)
    else:
        start = convert_date_bdte(start, options)

    nw = convert_date_bdte(date, options)
    if dc_convention and isinstance(dc_convention, str):
        date_diff = start.get_day_count(nw, dc_convention)
    else:
        date_diff = start.get_day_count(nw, options['control']["convention"])

    return date_diff

def calc_bdte_diff_int(upper, lower, options, dc_convention=None, dbg=False):
    ''' calculates dat difference and rounds to nearest year '''

    start = convert_date_bdte(lower, options)
    end = convert_date_bdte(upper, options)

    if dc_convention and dc_convention and isinstance(dc_convention, str):
        date_diff = start.get_day_count(end, dc_convention)
    elif options and 'control' in options.keys() and 'convention' in options['control']:
        date_diff = start.get_day_count(end, options['control']['convention'])
    else:
        date_diff = start.get_day_count(end, '30360')

    if 'control' in options.keys() and 'date_tol' in options['control'].keys():
        tol = options['control']['date_tol']
    else:
        tol = 0.15

    date_diff_final = adjust_diff_to_int(date_diff, tol)

    if dbg:
        print(start.to_date(), end.to_date(), date_diff_final)
    return date_diff_final

def adjust_diff_to_int(diff, tol):
    ''' adjustment code '''
    tol = (0.15 if np.isnan(tol) else tol)
    if  -1*tol < (diff - np.ceil(diff)) <= 0:
        diff_final = int(np.ceil(diff))
    elif  0 < diff - np.floor(diff) <= tol:
        diff_final = int(np.floor(diff))
    else:
        diff_final = int(np.floor(diff))
    return diff_final

def convert_period(period='Y'):
    '''converts JSON period into BusinessDate period '''
    if period.upper().startswith('M'):
        per = '1M'
    elif period.upper().startswith('Q'):
        per = '1Q'
    elif period.upper().startswith('S'):
        per = '6M'
    elif period.upper().startswith('W'):
        per = '1W'
    elif period.upper().startswith('D'):
        per = '1D'
    else:
        per = '1Y'

    return per

def calc_schedule(start, count, options, period='Y'):
    ''' Calculates schedules of payments dates based on strt and stop
    start: starting date
    count: number periods to generate
    period: type period to generate
    options: control dictionary
    '''

    strt_fnl = convert_date_bdte(start, options)

    if period.upper().startswith('M'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(months=count)
        per = '1M'
    elif period.upper().startswith('Q'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(quarters=count)
        per = '1Q'
    elif period.upper().startswith('S'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(months=count)
        per = '6M'
    elif period.upper().startswith('W'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(weeks=count)
        per = '1W'
    elif period.upper().startswith('D'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(days=count)
        per = '1D'
    else:
        stp_fnl = strt_fnl + bdte.BusinessPeriod(years=count)
        per = '1Y'

    sched = bdte.BusinessSchedule(strt_fnl, stp_fnl, per)
    if 'control' in options.keys() and 'date_adjust' in options['control'].keys() and\
            options['control']['date_adjust'] in ['follow', 'flw', 'modified']:
        sched.adjust(options['control']['date_adjust'])

    return sched

def constuct_schedule(dates, options, position, start, stop, count, period='Y'):
    ''' Calculates and construct schedule based on controls
    dates: array of dates
    options: control dictionary
    postion: location for first date (strt)
    start: starting date
    stop: ending date
    count: number periods to generate
    period: type period to generate
    '''
    base_dates = calc_schedule(start, count, options, period)

    if 'date_adjust' in options['control'].keys():
        adj = options['control']['date_adjust'].lower()
    else:
        adj = 'none'

    for loc, dte in zip(np.arange(0, len(base_dates)), base_dates):

        if adj == 'follow':
            if dte.is_business_day():
                date_final = dte.to_date()
            else:
                date_final = dte.adjust(convention=adj).to_date()

        elif  adj == 'final':
            dte2 = convert_date_bdte(stop, options)
            if loc == (len(base_dates)-1):
                if dte != dte2:
                    date_final = dte2.to_date()
                else:
                    date_final = dte.to_date()
            else:
                date_final = dte.to_date()
        else:
            date_final = dte.to_date()

        dates[position + loc] = dt.datetime(date_final.year, date_final.month, date_final.day,
                                            0, 0)
    return dates

def futures_rate(price):
    ''' calculates futures rate given futures price '''
    return 100.*(1. - price*0.01)

def inverse_futures_rates(rate):
    ''' calculates inverse futures rates (i.e. price) given rate '''
    return 100. - rate

def calc_yield_continuous(df, options):
    ''' calculates continuous yield based on previously calculated zero coupon '''
    if df is not None and isinstance(df, pd.DataFrame) and np.all(df.shape):
        df.loc[:, 'yield'] = -100.*np.log(df.loc[:, 'zero']) / df.loc[:, 'maturity']
    else:
        raise ValueError("df must be valid dataframe")

    return df

def calc_spot_simple(df, options):
    ''' calculates simple spot rate based on perviously calculated zero coupon '''
    if df is not None and isinstance(df, pd.DataFrame) and np.all(df.shape):
        df.loc[:, 'spot'] = 100.*(1. / df.loc[:, 'zero'] - 1.) / df.loc[:, 'maturity']
    else:
        raise ValueError("df must be valid dataframe")

    return df

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
                    df.at[itm[0], 'forward'] = 100./(
                        float(curr['maturity']) -
                        float(prev['maturity']))*(float(prev['zero'])/float(curr['zero']) - 1.)
                else:
                    df.at[itm[0], 'forward'] = 100./(float(curr['maturity']))*(
                        1.0 / float(curr['zero']) - 1.)
            prev = itm[1].copy()
            # print(type(prev))
    else:
        raise ValueError("df must be valid dataframe")

    return df

def build_arrays(options, dbg=False):
    ''' Constructs DataFrame elements '''
    if "count" in options.keys():
        cnt = options["count"]
    elif 'control' in options.keys() and "count" in options['control'].keys():
        cnt = options['control']["count"]

    if "columns" in options['control'].keys():
        rows = len(options['control']["columns"])
    df = np.zeros([cnt, rows])
    dates = np.repeat(dt.datetime.strptime(options["start_date"],
                                           options['control']["date_format"]), cnt)
    if dbg:
        print("Construction NP array size %i length %i" % (df.size, len(df)))

    return df, dates

def apply_yield_forward_calcs(df, options):
    ''' Calculates spot, yield and forwards based on provides zeros
    '''
    df = calc_yield_continuous(df, options)
    df = calc_spot_simple(df, options)
    df = calc_forward_rate(df, options)

    return df

def interpolate_instruments(result_position, result_date, position1, position2, options,
                            df, dates, dbg=False):
    ''' Simple linear interpolator that constructs interpolated instrument of same time
    result_position: location (numpy -- row) where results stored, determines location of date
    result_date: maturity date of instrument / rate -- determines weightings
    position1: location in df of left value used in interpolation
    position2: location in df of right valued used in interpolation
    options: controls dictionary
    df:     numpy containing interpolation inputs and result interpolation
    dates: date numpy 9.96400000e+01array containing dates (in dt.datetime)
    dbg: determines whether to output intermediate results from interpolation
    '''
    if int(df[position1][3]) == int(df[position2][3]):
        if "origin" in options['control']["columns"]:
            df[result_position][4] = load_types.INTERPOLATED.value

        if isinstance(result_date, str):
            dates[result_position] = dt.datetime.strptime(
                result_date, options['control']["date_format"])
        elif isinstance(result_date, bdte.BusinessDate):
            init_date = result_date.to_date()
            dates[result_position] = dt.datetime(init_date.year, init_date.month,
                                                 init_date.day, 0, 0)
        else:
            dates[result_position] = result_date

        df[result_position][0] = calc_bdte_diff(dates[result_position], options)

        dist = df[position2][0] - df[position1][0]
        q = (df[result_position][0] - df[position1][0]) / dist
        df[result_position][1] = (1- q)*df[position1][1] + q*df[position2][1]

        df[result_position][3] = df[position1][3]

        if int(df[position1][3]) == 1:
            df[result_position][2] = calc_libor_zero_coupon(
                df[result_position][1], df[result_position][0])

        elif int(df[position1][3]) == 2 or int(df[position1][3]) == 3:
            df[result_position][2] = calc_forward_zero_coupon(
                df[result_position][1], (df[result_position][0] - df[position1][0]),
                df[position1][2])

        if dbg:
            print("Interpolate Results: %d %d  %d  %i" % (
                df[result_position][0], df[result_position][1],
                df[result_position][2], int(df[result_position][3])))

    else:
        raise ValueError("(interpolate_instruments): Instrument Types MUST MATCH")

    return df, dates

def interpolate_price_swaps(rate, df, dates, options, positions=None,
                            pos_dates=None, ref_position=23, dbg=False):
    '''Interpolates swap rates and prices swaps iteratively
    rate: swap rate applied to final instrument
    df: np array populated by process for rows positions[0] -> position[1]
    dates: array of dates populated by function
    options: control dictionary
    positions: start & stop position for proceduce (populates posiitons[0] through
                positions[1]. Default = None (throws ValeuError)
    pos_dates:  date of first value and date for last value (used tp populate
                dates array. Defaults to None (throws ValueError)
    ref_position: position for first reference instrument
    dbg: debug indicator
    ex (1):  interpolate_price_swaps(3.0, df, dates, options, positions=[24, 28],
    ex (2):                        pos_dates=['2023-10-03', '2027-10-03'],
    ex (3):                        ref_position=23, dbg=False)
    '''
    if ref_position is None:
        ref_position = positions[0]-1

    if positions is not None and isinstance(positions, list) and\
            len(positions) >= 2:
        if pos_dates is not None and\
                isinstance(pos_dates, list) and len(pos_dates) >= 2:
            if dbg:
                print("positions & pos_dates confirm requirements")
        else:
            raise ValueError("Faulty pos_dates -- not None +  list and len(list) >=2")
    else:
        raise ValueError("Faulty positions -- not None + list + len(list) >= 2")

    df, dates = load_data_row(options, df, dates, position=positions[1], rate=rate,
                              date=pos_dates[1], typ='SWAP', ref_position=-2, dbg=dbg)

    dates = constuct_schedule(dates, options, positions[0], pos_dates[0], pos_dates[1],
                              (positions[1] - positions[0]), period='Y')

    for loc, dte in zip(np.arange(positions[0], positions[1], 1),
                        dates[positions[0]:positions[1]]):
        df, dates = interpolate_instruments(loc, dte, ref_position, positions[1],
                                            options, df, dates, dbg=dbg)

    for loc in np.arange(positions[0], positions[1]+1, 1):
        if loc == positions[0]:
            df[loc][2] = calc_swap_zero_coupon(df[loc][1], df, loc,
                                               ref_position=ref_position)
        else:
            df[loc][2] = calc_swap_zero_coupon(df[loc][1], df, loc,
                                               ref_position=(loc-1))

    return df, dates

def load_data_row(options, df=None, dates=None, position=0, rate=0.15, date='2012-10-02',
                  typ='LIBOR', ref_position=-1, dbg=True):
    ''' Loader -- operates based on typ '''
    if df is not None and isinstance(df, np.ndarray) and\
            int(df.size / len(df)) == len(options['control']["columns"]):
        if dbg:
            print("Fine Dimensions")
    else:
        df, dates = build_arrays(options, dbg=dbg)

    dates[position] = dt.datetime.strptime(date, options['control']["date_format"])
    if "origin" in options['control']["columns"]:
        df[position][4] = load_types.MARKET.value

    if typ.upper().startswith('LIBOR'):
        df[position][0] = calc_bdte_diff(dates[position], options)
        df[position][1] = rate
        df[position][2] = calc_libor_zero_coupon(rate, df[position][0])
        df[position][3] = rate_instruments.LIBOR.value
    elif typ.upper().startswith("FUTUR"):
        df[position][0] = calc_bdte_diff(dates[position], options)
        df[position][1] = futures_rate(rate)

        if 0 <= ref_position < position:
            df[position][2] = calc_forward_zero_coupon(
                df[position][1], (df[position][0] - df[ref_position][0]), df[ref_position][2])

        df[position][3] = rate_instruments.FUTURE.value
    elif typ.upper().startswith("SWAP"):
        df[position][0] = calc_bdte_diff(dates[position], options)
        df[position][1] = rate

        if 0 <= ref_position < position:
            df[position][2] = calc_swap_zero_coupon(rate, df, position, ref_position)

        df[position][3] = rate_instruments.SWAP.value
    else:
        raise ValueError("Type not supported")


    return df, dates

class short_rate_model():
    ''' base class corresponding to mean reverting short rate model '''

    def __init__(self, kappa, theta, sigma, r0):
        ''' constructor
        '''
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0


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
