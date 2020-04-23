''' Common pricing methods corresponding to Interest rate Instruments '''
import datetime as dt
#from collections import OrderedDict
import json
import os
import scipy as sci
import businessdate as bdte
import numpy as np
import pandas as pd
import interest_rate_utilities as intutil
import interest_rate_instruments as intrate


class curve_builder():
    ''' Class constructing applying exact mmethods:
    methods: (0): Bootsrap (1): Pseudo Inverse (2): Hilbert Space Cubic Spline
    '''

    def __init__(self, options, method=0, dbg=False):
        ''' Constructor Depends options dictionary '''
        if isinstance(options, str) and os.path.exists(options):
            with open(options, "r") as fp:
                self.options = json.load(fp)
            fp.close()
        elif isinstance(options, dict):
            self.options = options.copy()
        else:
            raise ValueError("Faulty -- options specification ")

        if 'control' not in self.options.keys():
            raise ValueError("defaults are not included in the options dictionary")

        self.curve_method = method
        self.swaps = {} #  OrderedDict()
        self.next_swap_key = 1
        self.dbg = dbg
        self.cf_dates = {} #  OrderedDict()
        self.names = {}
        self.cf_matrix = None
        self.cf_prices = None
        self.results = None
        self.count_instruments_and_dates()
        self.build_arrays()
        self.load_cf_matrices()
        if self.curve_method == 0:
            result = self.calc_exact_method0()
            if result:
                self.apply_yield_forward_calcs()

    def count_instruments_and_dates(self):
        ''' Counts totoal # of numbers loaded / interpolated by building process '''
        if 'instruments' in self.options.keys():
            for key, item in self.options["instruments"].items():
                if self.dbg:
                    print("Count Instr & Date: %s %s " % (key, item['date']))

                if item['type'].upper().startswith("FUTUR"):
                    self.append_instrument_date(item['date'], key, True)
                    key1 = "-".join([key, "reset"])
                    self.append_instrument_date(item['ref_date'], key1, False)

                elif item['type'].upper().startswith("SWAP"):
                    self.calc_next_swap_key(key)
                    self.swaps[key] = intrate.build_swap(key, item, self.options, dbg=False)
                    self.append_instrument_dates(key)
                else:
                    self.append_instrument_date(item['date'], key, True)
        else:
            raise ValueError("No Instruments found in dictionary")

        if 'interpolated_instruments' in self.options.keys():
            for key, item in self.options['interpolated_instruments'].items():
                self.append_instrument_date(item['date'], key, True)
        else:
            if self.dbg:
                print("Warning interpolated instruments found")

        if 'interpolated_swaps' in self.options.keys():
            self.append_swaps()
        else:
            if self.dbg:
                print("Warning interpolated swap found!!")

    def append_instrument_dates(self, instrument):
        ''' Appends series of dates for single instrument '''
        if self.swaps[instrument].instrument_type == intutil.rate_instruments.SWAP and\
                all(self.swaps[instrument].cash_flow_df.shape) > 0:
            for itm in self.swaps[instrument].cash_flow_df.index:
                if itm == self.swaps[instrument].reset:
                    key1 = self.options['control']['split'].join(
                        [self.swaps[instrument].name, "reset"])
                    key_add = bool('control' in self.options.keys() and 'append_reset' in
                                   self.options['control'].keys() and
                                   self.options['control']['append_reset'] > 0)

                    self.append_instrument_date(itm, key1, key_add)
                else:
                    if itm == self.swaps[instrument].maturity:
                        key1 = self.options['control']['split'].join(
                            [self.swaps[instrument].name, "CF+Notional"])
                        key_add = bool('control' in self.options.keys() and 'append_reset' in
                                       self.options['control'].keys() and
                                       self.options['control']['append_reset'] < 1)

                    else:
                        key1 = self.options['control']['split'].join(
                            [self.swaps[instrument].name, "CF"])
                        key_add = False
                    self.append_instrument_date(itm, key1, key_add)
        else:
            if self.dbg:
                print("Warning no instrument appended %s " % (instrument))

    def append_instrument_date(self, date, instrument, is_instrument=True):
        ''' Adds date instrument combination to list of dates '''
        if isinstance(date, bdte.BusinessDate):
            date_str = dt.datetime.strftime(date.to_date(), self.options['control']['date_format'])
        elif isinstance(date, (dt.date, dt.datetime)):
            date_str = dt.datetime.strftime(date, self.options['control']['date_format'])
        elif isinstance(date, str):
            dte = dt.datetime.strptime(date, self.options['control']['date_format'])
            date_str = dt.datetime.strftime(dte, self.options['control']['date_format'])

        if date_str in self.cf_dates.keys():
            self.cf_dates[date_str]['instruments'].append(instrument)
        else:
            self.cf_dates[date_str] = {'instruments': []}
            self.cf_dates[date_str]['instruments'].append(instrument)

        if is_instrument:
            key = (instrument if instrument.find(self.options['control']['split']) < 0 else
                   instrument.split(self.options['control']['split'])[0])

            if key not in self.names.keys():
                self.names[key] = date_str
            else:
                if self.dbg:
                    print("append_instrument_date %s %s" % (instrument, key))

    def append_swaps(self):
        ''' Appends SWAPs interpolated SWAPs '''
        self.next_swap_key = self.next_swap_key - 1

        for _, item in self.options['interpolated_swaps'].items():
            strt = intutil.convert_date_bdte(item['lower_date'], self.options)
            end = intutil.convert_date_bdte(item['upper']['date'], self.options)

            date_diff_dbl = intutil.calc_bdte_diff(end, self.options, strt)
            date_diff_final = intutil.calc_bdte_diff_int(
                end, strt, self.options, dbg=False)

            per = intutil.convert_period(item['upper']['frequency'])

            sched = bdte.BusinessSchedule(strt, end, per)
            if 'control' in self.options.keys() and\
                    'date_adjust' in self.options['control'].keys() and\
                    self.options['control']['date_adjust'] in ['follow']:
                # key adjusts sched for business date -- print("Here")
                sched.adjust(self.options['control']['date_adjust'])

            if self.dbg:
                print("## ", self.next_swap_key, strt, end, date_diff_final)

            self.next_swap_key = (self.next_swap_key + date_diff_final)
            key = "".join(["SWAP", str(self.next_swap_key)])
            # self.calc_next_swap_key(key)
            self.swaps[key] = intrate.build_swap(key, item['upper'], self.options, dbg=False)
            self.append_instrument_dates(key)
            swap_key = self.find_swap(min(sched))
            swap_strt = int(swap_key.upper().split("AP")[1])

            for loc, dte in zip(np.arange(0, len(sched)), sched):
                if  min(sched) < dte < max(sched):
                    new_swap_dict = item['upper'].copy()
                    new_swap_dict['date'] = dte
                    date_diff_dbl2 = intutil.calc_bdte_diff(dte, self.options, strt)
                    dbl2 = date_diff_dbl2 / date_diff_dbl
                    new_swap_dict['rate'] = sci.interp(dbl2, xp=[0.0, 1.0], fp=[
                        self.swaps[swap_key].r_swap, self.swaps[key].r_swap])
                    new_swap_name = "".join(["SWAP", str(swap_strt + loc)])
                    self.swaps[new_swap_name] = intrate.build_swap(
                        new_swap_name, new_swap_dict, self.options, dbg=False)
                    self.append_instrument_dates(new_swap_name)


    def calc_next_swap_key(self, key=None):
        ''' calculates next SWAP key used as primary key '''
        if key is None or not isinstance(key, str):
            self.next_swap_key = self.next_swap_key + 1
        else:
            swap_key = int(key.split("AP")[1])
            if swap_key == self.next_swap_key:
                self.next_swap_key = self.next_swap_key + 1
            elif  swap_key > self.next_swap_key:
                self.next_swap_key = "".join(["SWAP", str(swap_key + 1)])

    def calc_schedule(self, start, stop, reference_date, count, period='Y'):
        ''' Calculates schedules of payments dates based on strt and stop
        dates: array of dates
        options: control dictionary
        postion: location for first date (strt)
        start: starting date
        stop: ending date
        count: number periods to generate
        periods: type period to generate
        '''

        base_dates = intutil.calc_schedule(start, count, self.options, period)

        adj = (self.options['control']['date_adjust'].lower()
               if 'date_adjust' in self.options['control'].keys() else
               'none')
        prev = ''

        for loc, dte in zip(np.arange(0, count+1), base_dates):

            if adj == 'final':
                if isinstance(stop, str):
                    dte2 = bdte.BusinessDate(dt.datetime.strptime(
                        stop, self.options['control']['date_format']))
                else:
                    dte2 = stop

                if loc == (len(base_dates)-1):
                    if dte != dte2:
                        date_final = dte2.to_date()
                    else:
                        date_final = dte.to_date()
                else:
                    date_final = dte.to_date()
            else:
                date_final = dte.to_date()

            keyname = "".join(["SWAP", str(self.next_swap_key)])
            self.append_instrument_date(date_final, keyname, True)
            key1 = "_".join([keyname, "reset"])

            if loc == 0:
                self.append_instrument_date(reference_date, key1, False)
            else:
                self.append_instrument_date(prev, key1, False)

            self.calc_next_swap_key()

            prev = date_final

    def find_swap(self, date):
        ''' searches swap dictionary for swap having date as maturity '''
        key_fnl = None
        for key, item in self.swaps.items():
            if item.maturity == date:
                key_fnl = key
                break
        return key_fnl

    def build_arrays(self):
        ''' Constructs DataFrame elements '''
        instrument_cnt = len(self.names)
        rows = len(self.cf_dates)
        dates = []
        for itm in self.cf_dates:
            if isinstance(itm, (dt.date, bdte.BusinessDate)):
                dates.append(dt.datetime(year=itm.year, month=itm.month, day=itm.day, hour=0,
                                         minute=0))
            else:
                dates.append(dt.datetime.strptime(itm, self.options['control']['date_format']))

        if self.cf_matrix is None:
            mtrx = np.zeros([rows, instrument_cnt])
            dates.sort()
            names_sorted = sorted(self.names, key=lambda x: self.names[x])
            self.cf_matrix = pd.DataFrame(mtrx, index=dates, columns=names_sorted)

        if self.dbg:
            print("Construction NP array shape %i length %i" % (self.cf_matrix.shape[0],
                                                                self.cf_matrix.shape[1]))

        if self.cf_prices is None and instrument_cnt > 0:
            self.cf_prices = pd.Series(np.zeros([instrument_cnt]), index=names_sorted)

        if self.results is None:
            if "columns" in self.options['control'].keys():
                rows = len(self.options['control']["columns"])
                mtrx = np.zeros([instrument_cnt, rows])
                self.results = pd.DataFrame(mtrx, index=names_sorted,
                                            columns=self.options['control']['columns'])

            if self.dbg:
                print("Construction NP array shape %i length %i" % (self.results.shape[0],
                                                                    self.results.shape[1]))

    def load_cf_matrices(self):
        ''' Loads all CF elements '''
        if self.cf_matrix is not None and self.cf_prices is not None and\
                self.results is not None:
            for item in self.cf_matrix.index:
                dte = self.construct_date_str(item)
                if dte in self.cf_dates.keys():
                    inst = self.determine_instrument_name(dte)
                    if inst in self.options['instruments'].keys():
                        if inst.upper().startswith("FORWA") or inst.upper().startswith("FUTURE"):
                            self.load_data_row(
                                position=inst, rate=self.options['instruments'][inst]['rate'],
                                date=self.options['instruments'][inst]['date'],
                                reference_date=self.options['instruments'][inst]['ref_date'],
                                typ=inst)
                        else:
                            self.load_data_row(position=inst,
                                               rate=self.options['instruments'][inst]['rate'],
                                               date=self.options['instruments'][inst]['date'],
                                               typ=inst)

                        self.results.loc[inst, 'loaded'] = 1
                    elif inst.upper().startswith('SWAP') and inst in self.swaps.keys():
                        self.load_data_row(
                            position=inst, rate=self.swaps[inst].r_swap,
                            date=self.swaps[inst].maturity,
                            reference_date=self.swaps[inst].reset,
                            typ=inst)
                    else:
                        if self.dbg:
                            print("Warning (instrument) %s not found" % (inst))


                    if self.dbg:
                        print(item, inst, str(bool(self.results.loc[inst, 'loaded'] > 0)))
                else:
                    print("Warning (date) %s not found" % (dte))
            if 'interpolated_instruments' in self.options.keys():
                for key, item in  self.options['interpolated_instruments'].items():
                    self.interpolate_instruments(key, item)
        else:
            raise ValueError("Missing class elements!!!")

    def determine_instrument_name(self, date, instrument=True):
        ''' Calculates instrument name based on provided date (string, same format cf_dates)'''
        inst = False
        if isinstance(instrument, bool) and instrument:
            for itm in self.cf_dates[date]['instruments']:
                if not itm.upper().endswith("RESET") and not itm.upper().endswith("REFERENCE") and\
                        not itm.upper().endswith("CF"):
                    if itm.find(self.options['control']['split']) >= 0:
                        inst = itm.split(self.options['control']['split'])[0]
                    else:
                        inst = itm
                    break
        elif isinstance(instrument, str) and instrument:
            inst = instrument in self.cf_dates[date]['instruments']
        else:
            if self.dbg:
                print("Faulty application of determine_instrument_name")

        return inst


    def construct_date_str(self, val):
        ''' Constructs date as string from object '''
        day = (str(val.day) if val.day > 9 else ("0" + str(val.day)))
        month = (str(val.month) if val.month > 9 else ("0" + str(val.month)))
        return self.options['control']['split'].join([str(val.year), month, day])

    def load_data_row(self, position='LIBOR1', rate=0.15, date='2012-10-02',
                      reference_date='2012-07-02', typ='LIBOR', origin=None):
        ''' Loader -- operates based on typ '''

        if "origin" in self.options['control']["columns"] and origin is None:
            self.results.loc[position]['origin'] = intutil.load_types.MARKET.value
        elif "origin" in self.options['control']["columns"] and\
                isinstance(origin, (int, float)):
            self.results.loc[position]['origin'] = origin

        mat = intutil.calc_bdte_diff(date, self.options)

        if typ.upper().startswith('LIBOR'):
            self.results.loc[position]['maturity'] = mat
            self.results.loc[position]['rate'] = rate
            self.results.loc[position]['type'] = intutil.rate_instruments.LIBOR.value

            # calc zero replaced with matrix solution (0) / pseudo inverse (1)
            self.results.loc[position, 'loaded'] = 1
            # calc zero replaced with matrix solution (0) / pseudo inverse (1)
            self.cf_matrix.loc[date][position] = (1 + 0.01*mat*rate)
            self.cf_prices[position] = 1.
        elif typ.upper().startswith("FUTUR"):
            self.results.loc[position]['maturity'] = mat
            self.results.loc[position]['rate'] = intutil.futures_rate(rate)
            self.results.loc[position]['type'] = intutil.rate_instruments.FUTURE.value

            # calc zero replaced with matrix solution (0) / pseudo inverse (1)
            self.cf_matrix.loc[reference_date][position] = -1.
            refereposition = self.determine_instrument_name(reference_date)

            self.cf_matrix.loc[date][position] = (1. +
                                                  (self.results.loc[position]['maturity'] -
                                                   self.results.loc[refereposition]['maturity'])
                                                  *0.01*self.results.loc[position]['rate'])

            self.results.loc[position, 'loaded'] = 1
        elif typ.upper().startswith("SWAP"):
            self.results.loc[position]['maturity'] = mat
            self.results.loc[position]['rate'] = rate
            self.results.loc[position]['type'] = intutil.rate_instruments.SWAP.value
            for loc, val in self.swaps[position].cash_flow_df.iterrows():
                self.cf_matrix.loc[loc, position] = val['CF']

            self.results.loc[position, 'loaded'] = 1
        else:
            raise ValueError("Type not supported")

    def interpolate_instruments(self, result_position, item_dict):
        ''' Simple linear interpolator that constructs interpolated instrument of same time
        result_position: location (numpy -- row) where results stored, determines location of date
        result_date: maturity date of instrument / rate -- determines weightings
        position1: location in df of left value used in interpolation
        position2: location in df of right valued used in interpolation
        '''
        if self.results.loc[item_dict['reference_positions'][0]]['maturity'] <=\
                self.results.loc[item_dict['reference_positions'][1]]['maturity']:
            position1 = item_dict['reference_positions'][0]
            position2 = item_dict['reference_positions'][1]
        else:
            position2 = item_dict['reference_positions'][0]
            position1 = item_dict['reference_positions'][1]

        if int(self.results.loc[position1]['type']) == int(self.results.loc[position2]['type']):
            if 'type' in item_dict.keys():
                typ = item_dict['type']
            else:
                typ = self.options['instruments'][position1]['type']

            mat = intutil.calc_bdte_diff(item_dict['date'], self.options)

            dist = self.results.loc[position2]['maturity'] - self.results.loc[position1]['maturity']
            q = (mat - self.results.loc[position1]['maturity']) / dist

            rate = (1- q)*self.results.loc[position1]['rate'] +\
                    q*self.results.loc[position2]['rate']

            if result_position.upper().startswith('LIBOR'):
                self.load_data_row(position=result_position, rate=rate, date=item_dict['date'],
                                   typ='LIBOR', origin=intutil.load_types.INTERPOLATED.value)
            else:
                if position1 in self.options['instruments'].keys():
                    ref_date = self.options['instruments'][position1]['date']

                    self.load_data_row(position=result_position, rate=rate, date=item_dict['date'],
                                       reference_date=ref_date, typ=typ,
                                       origin=intutil.load_types.INTERPOLATED.value)
                else:
                    print("Warning no item %s loaded" % (result_position))

            if self.dbg:
                print("Interpolate Results: %d %d  %d  %i" % (
                    self.results.loc[result_position][0], self.results.loc[result_position][1],
                    self.results.loc[result_position][2],
                    int(self.results.loc[result_position][3])))

        else:
            raise ValueError("(interpolate_instruments): Instrument Types MUST MATCH")

    def calc_exact_method0(self):
        ''' implements eaxct method with complete NxN matrix '''
        res = False
        if isinstance(self.cf_matrix, pd.DataFrame) and all(self.cf_matrix.shape) > 0 and\
                isinstance(self.cf_prices, pd.Series) and self.cf_prices.size > 0:
            zeros = sci.linalg.solve(self.cf_matrix.transpose(), self.cf_prices)

            if all(np.logical_not(np.isnan(zeros))) and\
                    np.logical_and(all(zeros > -0.00005), all(zeros < 1.001)):
                res = True

                for loc, val in zip(self.results.index, zeros):
                    self.results.loc[loc, 'zero'] = val
        else:
            if self.dbg:
                print("Warning -- zeros NOT calculated")
        return res

    def apply_yield_forward_calcs(self):
        ''' Calculates spot, yield and forwards based on provides zeros
        '''
        self.results = intutil.calc_yield_continuous(self.results, self.options)
        self.results = intutil.calc_spot_simple(self.results, self.options)
        self.results = intutil.calc_forward_rate(self.results, self.options)
