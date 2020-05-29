''' Common pricing methods corresponding to Interest rate Instruments '''
import datetime as dt
import businessdate as bdte
import numpy as np
import interest_rate_base as intbase
import interest_rate_dates as intdate


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
    df = intbase.calc_yield_continuous(df)
    df = intbase.calc_spot_simple(df)
    df = intbase.calc_forward_rate(df, options)

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
            df[result_position][4] = intbase.load_types.INTERPOLATED.value

        if isinstance(result_date, str):
            dates[result_position] = dt.datetime.strptime(
                result_date, options['control']["date_format"])
        elif isinstance(result_date, bdte.BusinessDate):
            init_date = result_date.to_date()
            dates[result_position] = dt.datetime(init_date.year, init_date.month,
                                                 init_date.day, 0, 0)
        else:
            dates[result_position] = result_date

        df[result_position][0] = intdate.calc_bdte_diff(dates[result_position], options)

        dist = df[position2][0] - df[position1][0]
        q = (df[result_position][0] - df[position1][0]) / dist
        df[result_position][1] = (1- q)*df[position1][1] + q*df[position2][1]

        df[result_position][3] = df[position1][3]

        if int(df[position1][3]) == 1:
            df[result_position][2] = intbase.calc_libor_zero_coupon(
                df[result_position][1], df[result_position][0])

        elif int(df[position1][3]) == 2 or int(df[position1][3]) == 3:
            df[result_position][2] = intbase.calc_forward_zero_coupon(
                df[result_position][1], (df[result_position][0] - df[position1][0]),
                df[position1][2])

        if dbg:
            print("Interpolate Results: %d %f (%f) %f  %d" % (
                df[result_position][0], df[result_position][1],
                (df[result_position][0] - df[position1][0]),
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

    dates = intdate.constuct_schedule(dates, options, positions[0], pos_dates[0], pos_dates[1],
                                      (positions[1] - positions[0]), period='Y')

    for loc, dte in zip(np.arange(positions[0], positions[1], 1),
                        dates[positions[0]:positions[1]]):
        df, dates = interpolate_instruments(loc, dte, ref_position, positions[1],
                                            options, df, dates, dbg=dbg)

    for loc in np.arange(positions[0], positions[1]+1, 1):
        if loc == positions[0]:
            df[loc][2] = intbase.calc_swap_zero_coupon(df[loc][1], df, loc,
                                                       ref_position=ref_position)
        else:
            df[loc][2] = intbase.calc_swap_zero_coupon(df[loc][1], df, loc,
                                                       ref_position=(loc-1))

    return df, dates

def load_data_row(options, df=None, dates=None, position=0, rate=0.15, date='2012-10-02',
                  typ='LIBOR', ref_position=-1, dbg=True):
    ''' DEPRECATED -- see curve constructor
    Loader -- operates based on typ
    '''
    if df is not None and isinstance(df, np.ndarray) and\
            int(df.size / len(df)) == len(options['control']["columns"]):
        if dbg:
            print("Fine Dimensions")
    else:
        df, dates = build_arrays(options, dbg=dbg)

    dates[position] = dt.datetime.strptime(date, options['control']["date_format"])
    if "origin" in options['control']["columns"]:
        df[position][4] = intbase.load_types.MARKET.value

    if typ.upper().startswith('LIBOR'):
        df[position][0] = intdate.calc_bdte_diff(dates[position], options)
        df[position][1] = rate
        df[position][2] = intbase.calc_libor_zero_coupon(rate, df[position][0])
        df[position][3] = intbase.rate_instruments.LIBOR.value
    elif typ.upper().startswith("FUTUR"):
        df[position][0] = intdate.calc_bdte_diff(dates[position], options)
        df[position][1] = intbase.futures_rate(rate)

        if 0 <= ref_position < position:
            df[position][2] = intbase.calc_forward_zero_coupon(
                df[position][1], (df[position][0] - df[ref_position][0]), df[ref_position][2])

        df[position][3] = intbase.rate_instruments.FUTURE.value
    elif typ.upper().startswith("SWAP"):
        df[position][0] = intdate.calc_bdte_diff(dates[position], options)
        df[position][1] = rate

        if 0 <= ref_position < position:
            df[position][2] = intbase.calc_swap_zero_coupon(rate, df, position, ref_position)

        df[position][3] = intbase.rate_instruments.SWAP.value
    else:
        raise ValueError("Type not supported")


    return df, dates
