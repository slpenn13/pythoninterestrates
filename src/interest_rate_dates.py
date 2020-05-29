''' Common pricing methods corresponding to Interest rate Instruments '''
import datetime as dt
import businessdate as bdte
import numpy as np


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
        diff_final = int(np.ceil(diff))
    return diff_final

def adjust_date_bd_convention(date, options, dbg=False):
    ''' adjusts dates based on business date convention '''
    if isinstance(date, bdte.BusinessDate):
        date_fnl = date
    else:
        if dbg:
            print("Warning -- date is being converted")
        date_fnl = convert_date_bdte(date, options)

    if not date_fnl.is_business_day() and\
            'date_adjust' in options['control'].keys() and\
            options['control']['date_adjust'] in ['follow', 'flw', 'modified']:
        date_fnl = date_fnl.adjust(options['control']['date_adjust'])

    return date_fnl

def convert_period_to_year(period, options):
    ''' converts period / frequency to year (float)'''
    if period.upper().startswith('M'):
        per = 1/12.
    elif period.upper().startswith('Q'):
        per = 0.25
    elif period.upper().startswith('S'):
        per = 0.50
    elif period.upper().startswith('W'):
        per = 1/52.
    elif period.upper().startswith('D'):
        if isinstance(options, dict) and 'control' in options.keys() and\
                'convention' in options['control'].keys():
            if options['control']['convention'].endswith('365'):
                per = 1/365.
            else:
                per = 1/360.
        else:
            per = 1/360.0
    elif period.upper().startswith('Y'):
        per = 1.0
    else:
        per = 0.0

    return per


def convert_period(period='Y'):
    '''converts JSON period into BusinessDate period '''
    if period.upper().startswith('M'):
        per = bdte.BusinessPeriod(months=1)
    elif period.upper().startswith('Q'):
        per = bdte.BusinessPeriod(quarters=1)
    elif period.upper().startswith('S'):
        per = bdte.BusinessPeriod(months=6)
    elif period.upper().startswith('W'):
        per = bdte.BusinessPeriod(weeks=1)
    elif period.upper().startswith('D'):
        per = bdte.BusinessPeriod(days=1)
    else:
        per = bdte.BusinessPeriod(years=1)

    return per

def convert_period_count(count, periods='Y'):
    ''' converts JSON period count plus num periods '''
    if periods.upper().startswith('M'):
        per = 12*count
    elif periods.upper().startswith('Q'):
        per = 4*count
    elif periods.upper().startswith('S'):
        per = 12*count
    elif periods.upper().startswith('W'):
        per = 52*count
    elif periods.upper().startswith('D'):
        per = 365*count
    else:
        per = count

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
        per = bdte.BusinessPeriod(months=1)
    elif period.upper().startswith('Q'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(quarters=count)
        per = bdte.BusinessPeriod(quarters=1)
    elif period.upper().startswith('S'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(months=count)
        per = bdte.BusinessPeriod(months=6)
    elif period.upper().startswith('W'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(weeks=count)
        per = bdte.BusinessPeriod(weeks=1)
    elif period.upper().startswith('D'):
        stp_fnl = strt_fnl + bdte.BusinessPeriod(days=count)
        per = bdte.BusinessPeriod(days=1)
    else:
        stp_fnl = strt_fnl + bdte.BusinessPeriod(years=count)
        per = bdte.BusinessPeriod(years=1)

    # sched = bdte.BusinessSchedule(strt_fnl, stp_fnl, per)
    sched = bdte.BusinessSchedule(strt_fnl, stp_fnl, per)
    if  options is not None and 'control' in options.keys() and\
            'date_adjust' in options['control'].keys() and\
            options['control']['date_adjust'] in ['follow', 'flw', 'modified']:
        sched.adjust(options['control']['date_adjust'])

    return sched

def calc_schedule_list(items, options):
    ''' calculates vectors of bdte.BusinessDate (sorted) a matrix Nx2'''
    res_date = []
    cnt = len(items)
    res_values = np.zeros([cnt, 2])
    for itm in items:
        res_date.append(convert_date_bdte(itm, options))

    res_date_final = sorted(res_date)

    for loc, itm in zip(np.arange(0, cnt), res_date_final):
        res_values[loc][0] = calc_bdte_diff(itm, options)
        if loc == 0:
            res_values[loc][1] = res_values[loc][0]
        else:
            res_values[loc][1] = res_values[loc][0] - res_values[loc-1][0]

    return res_date_final, res_values

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

def generate_schedule_dict(start=None, period='Y', count=0, date_format="%Y-%m-%d",
                           convention='30360', date_adjust='follow'):
    ''' helper function: generates system compliant schedule scpecification '''

    res_dict = {}
    res_dict['control'] = {}
    res_dict['control']['convention'] = convention
    res_dict['control']['date_adjust'] = date_adjust
    res_dict['control']['date_format'] = date_format

    if start is None:
        base_date = dt.date.today()
    elif isinstance(start, (dt.date, dt.datetime, str, bdte.BusinessDate)):
        base_date = start
    else:
        raise ValueError("Faulty Starting Date")

    strt = convert_date_bdte(base_date, res_dict)
    res_dict['frequency'] = period
    if count and isinstance(count, (int, float)):
        res_dict['count'] = int(count)
        res_dict['start'] = strt
    elif count and isinstance(count, (dt.date, dt.datetime, str, bdte.BusinessDate)):
        res_dict['end'] = convert_date_bdte(count, res_dict)
        res_dict['start'] = strt
    elif count and isinstance(count, list) and isinstance(count[0], bdte.BusinessDate):
        res_dict['schedule'] = []
        if strt not in count:
            res_dict['schedule'].append(strt)
        res_dict['schedule'].extend(count)

    return res_dict
