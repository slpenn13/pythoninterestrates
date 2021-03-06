''' Common pricing methods corresponding to Interest rate Instruments '''
import datetime as dt
#from collections import OrderedDict
import json
import os
import scipy as sci
import businessdate as bdte
import numpy as np
import pandas as pd
import interest_rate_instruments as intrate
import interest_rate_base as intbase
import interest_rate_dates as intdate
import interest_rate_discount as intdisc
import interest_rate_swap_convenience as swapconv


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
        # self.swap_t0_equal = swap_t0_equal
        self.instruments = {} #  OrderedDict()
        self.next_swap_key = 1
        self.dbg = dbg
        self.cf_dates = {} #  OrderedDict()
        self.names = {} #  instrument names
        self.cf_matrix = None
        self.cf_prices = None
        self.results = None
        self.deltas = None
        self.zeros = None
        self.count_instruments_and_dates()
        self.build_arrays()
        self.load_cf_results()
        self.load_cf_matrices()
        if self.curve_method == 0:
            result = self.calc_exact_method0()
            if result:
                self.apply_yield_forward_calcs()
        elif self.curve_method == 1:
            result = self.calc_exact_method1()
            if result:
                self.apply_yield_forward_calcs()
        else:
            if self.dbg:
                print("Warning -- no method ID %d exists" % (self.curve_method))

    def count_instruments_and_dates(self):
        ''' Counts totoal # of numbers loaded / interpolated by building process '''
        if 'instruments' in self.options.keys():
            for key, item in self.options["instruments"].items():
                if self.dbg:
                    print("Count Instr & Date: %s %s " % (key, item['date']))

                if item['type'].upper().startswith("FUTUR"):
                    self.instruments[key] = intrate.interest_rate_future(
                        key, item['rate'], item['date'], item['ref_date'],
                        frequency=item['frequency'], options=self.options, dbg=False)

                    self.append_instrument_date(item['date'], key, True)
                    key1 = "-".join([key, "reset"])
                    self.append_instrument_date(item['ref_date'], key1, False)

                elif item['type'].upper().startswith("SWAP"):
                    self.calc_next_swap_key(key)
                    self.instruments[key] = swapconv.build_swap(key, item, self.options, dbg=False)
                    self.append_instrument_dates_swap(key)
                elif item['type'].upper().startswith("FIXEDCOUPONBO"):
                    init_cpn = intbase.fixed_coupon(
                        coupon=item['coupon'], frequency=item['frequency'],
                        convention=self.options['control']['date_adjust'], adjust=False)
                    self.instruments[key] = intrate.fixed_coupon_bond(
                        key, item['nextcoupon'], item['date'], self.options,
                        init_cpn, princ=item['notional'], price=item['price'],
                        dated=self.options['start_date'])

                    self.append_instrument_dates_bond(key)
                elif item['type'].upper().startswith("ZERO"):
                    opt2 = self.options.copy()
                    dflt_day_cnt = (self.options['control']['convention'] if 'control' in
                                    self.options.keys() and 'convention' in
                                    self.options['control'].keys() else '30360')

                    opt2['control']['convention'] = (item['day_count'] if 'day_copunt' in
                                                     item.keys() else dflt_day_cnt)
                    self.instruments[key] = intrate.fi_instrument(
                        key, opt2['start_date'], item['date'], princ=100.0, frequency=None,
                        options=opt2)

                    if 'price' in item.keys():
                        self.instruments[key].generate_cf(price=item['price'])

                    self.append_instrument_dates_bond(key)
                else:
                    self.append_instrument_date(item['date'], key, True)
        else:
            raise ValueError("No Instruments found in dictionary")

        if 'interpolated_instruments' in self.options.keys():
            for key, item in self.options['interpolated_instruments'].items():
                self.append_instrument_date(item['date'], key, True)
        else:
            if self.curve_method == 0 and self.dbg:
                print("Warning NO interpolated instruments found")

        if 'interpolated_swaps' in self.options.keys():
            self.append_swaps()
        else:
            if self.curve_method == 0 and self.dbg:
                print("Warning NO interpolated swap found!!")

    def append_instrument_dates_swap(self, instrument):
        ''' Appends series of dates for single instrument '''
        if self.instruments[instrument].instrument_type == intbase.rate_instruments.SWAP and\
                all(self.instruments[instrument].cash_flow_df.shape) > 0:
            for itm in self.instruments[instrument].cash_flow_df.index:
                if itm == self.instruments[instrument].reset:
                    key1 = self.options['control']['split'].join(
                        [self.instruments[instrument].name, "reset"])
                    key_add = bool('control' in self.options.keys() and 'append_reset' in
                                   self.options['control'].keys() and
                                   self.options['control']['append_reset'] > 0)

                    self.append_instrument_date(itm, key1, key_add)
                else:
                    if itm == self.instruments[instrument].maturity:
                        key1 = self.options['control']['split'].join(
                            [self.instruments[instrument].name, "CF+Notional"])
                        key_add = bool('control' in self.options.keys() and 'append_reset' in
                                       self.options['control'].keys() and
                                       self.options['control']['append_reset'] < 1)

                    else:
                        key1 = self.options['control']['split'].join(
                            [self.instruments[instrument].name, "CF"])
                        key_add = False
                    self.append_instrument_date(itm, key1, key_add)
        else:
            if self.dbg:
                print("Warning no instrument appended %s " % (instrument))

    def append_instrument_dates_bond(self, instrument):
        ''' Appends series of bond payment dates '''
        if (self.instruments[instrument].instrument_type ==\
                    intbase.rate_instruments.FIXED_RATE_BOND or\
                    self.instruments[instrument].instrument_type ==\
                    intbase.rate_instruments.ZERO_COUPON) and\
                all(self.instruments[instrument].cash_flow_df.shape) > 0:

            for indx, row in self.instruments[instrument].cash_flow_df.iterrows():
                if abs(row['CF']) > 0.0:
                    if indx == self.instruments[instrument].maturity:
                        key1 = self.options['control']['split'].join(
                            [self.instruments[instrument].name, "CF+Notional"])
                        key_add = True
                    else:
                        key1 = self.options['control']['split'].join(
                            [self.instruments[instrument].name, "CF"])
                        key_add = False
                    self.append_instrument_date(indx, key1, key_add)
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
            strt = intdate.convert_date_bdte(item['lower_date'], self.options)
            end = intdate.convert_date_bdte(item['upper']['date'], self.options)

            per = intdate.convert_period(item['upper']['frequency'])

            sched = bdte.BusinessSchedule(strt, end, per)
            if 'control' in self.options.keys() and\
                    'date_adjust' in self.options['control'].keys() and\
                    self.options['control']['date_adjust'] in ['follow', 'flw', 'modified']:
                # key adjusts sched for business date -- print("Here")
                sched.adjust(self.options['control']['date_adjust'])


            date_diff_final = intdate.calc_bdte_diff_int(
                end, strt, self.options, dbg=False)

            if self.dbg:
                print("## ", self.next_swap_key, strt, end, date_diff_final)

            self.next_swap_key = (self.next_swap_key + date_diff_final)
            if 'control' in self.options.keys() and 'date_adjust' in\
                    self.options['control'].keys():
                if not strt.is_business_day():
                    strt.adjust(self.options['control']['date_adjust'])
                if not end.is_business_day():
                    end.adjust(self.options['control']['date_adjust'])

            date_diff_dbl = intdate.calc_bdte_diff(end, self.options, strt)
            key = "".join(["SWAP", str(self.next_swap_key)])
            # self.calc_next_swap_key(key)
            self.instruments[key] = swapconv.build_swap(key, item['upper'], self.options, dbg=False)
            self.append_instrument_dates_swap(key)
            swap_key = self.find_swap(min(sched))
            swap_strt = int(swap_key.upper().split("AP")[1])

            for loc, dte in zip(np.arange(0, len(sched)), sched):
                if  min(sched) < dte < max(sched):
                    new_swap_dict, new_swap_name = swapconv.generate_swap_dictionary(
                        self.options, maturity=dte, swap_start=strt,
                        swapid=(swap_strt + loc), reference="SWAP1",
                        rates=[[0.0, date_diff_dbl],
                               [self.instruments[swap_key].r_swap, self.instruments[key].r_swap]],
                        dbg=False)

                    self.instruments[new_swap_name] = swapconv.build_swap(
                        new_swap_name, new_swap_dict, self.options, dbg=False)

                    self.append_instrument_dates_swap(new_swap_name)

    def calc_next_swap_key(self, key=None):
        ''' calculates next SWAP key used as primary key '''
        if key is None or not isinstance(key, str):
            self.next_swap_key = self.next_swap_key + 1
        else:
            swap_key = int(key.split("AP")[1])
            if swap_key == self.next_swap_key:
                self.next_swap_key = self.next_swap_key + 1
            elif  swap_key > self.next_swap_key:
                self.next_swap_key = swap_key + 1

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

        base_dates = intdate.calc_schedule(start, count, self.options, period)

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
        for key, item in self.instruments.items():
            if item.instrument_type == intbase.rate_instruments.SWAP and item.maturity == date:
                key_fnl = key
                break
        return key_fnl

    def build_arrays(self):
        ''' Constructs DataFrame elements '''
        instrument_cnt = len(self.names)
        dates = []
        for itm in self.cf_dates:
            if self.calc_date_inclusion(itm):
                if isinstance(itm, (dt.date, bdte.BusinessDate)):
                    dates.append(dt.datetime(year=itm.year, month=itm.month, day=itm.day, hour=0,
                                             minute=0))
                else:
                    dates.append(dt.datetime.strptime(itm, self.options['control']['date_format']))
            else:
                if self.dbg:
                    print("build_arrays -- excluding %s" % (itm))


        rows = len(dates)
        if self.cf_matrix is None: # cf_matrix -- # dates X # instruments
            mtrx = np.zeros([rows, instrument_cnt])
            dates.sort()
            names_sorted = sorted(self.names, key=lambda x: self.names[x])
            self.cf_matrix = pd.DataFrame(mtrx, index=dates, columns=names_sorted)

        if self.dbg:
            print("Construction NP array shape %i length %i" % (self.cf_matrix.shape[0],
                                                                self.cf_matrix.shape[1]))

        if self.cf_prices is None and instrument_cnt > 0: # # instruments
            self.cf_prices = pd.Series(np.zeros([instrument_cnt]), index=names_sorted)

        if self.results is None: # # instruments X column count
            if "columns" in self.options['control'].keys():
                rows = len(self.options['control']["columns"])
                mtrx = np.zeros([instrument_cnt, rows])
                self.results = pd.DataFrame(mtrx, index=names_sorted,
                                            columns=self.options['control']['columns'])

            if self.dbg:
                print("Construction NP array shape %i length %i" % (self.results.shape[0],
                                                                    self.results.shape[1]))

    def load_cf_results(self):
        ''' Loads results elemt upto (but not including) zeros '''
        if self.results is not None and self.cf_prices is not None:
            for item in self.cf_matrix.index:
                dte = self.construct_date_str(item)
                if dte in self.cf_dates.keys():
                    inst = self.determine_instrument_name(dte)
                    if not isinstance(inst, bool) and inst in self.options['instruments'].keys():
                        if inst.upper().startswith("FORWA") or inst.upper().startswith("FUTURE"):
                            self.load_data_row(
                                position=inst, rate=self.instruments[inst].rate,
                                date=self.instruments[inst].maturity, typ=inst)
                        elif inst.upper().startswith('SWAP'):
                            self.load_data_row(
                                position=inst, rate=self.instruments[inst].r_swap,
                                date=self.instruments[inst].maturity,
                                typ=inst, origin=int(self.instruments[inst].is_market_quote.value))
                        elif not isinstance(inst, bool) and inst.upper().startswith('BOND') and\
                                inst in self.instruments.keys():
                            self.load_data_row(
                                position=inst, rate=self.instruments[inst].coupon.coupon,
                                date=self.instruments[inst].maturity,
                                typ="FIXEDCOUPONBOND", origin=1)
                        else:
                            self.load_data_row(position=inst,
                                               rate=self.options['instruments'][inst]['rate'],
                                               date=self.options['instruments'][inst]['date'],
                                               typ=inst)

                    elif not isinstance(inst, bool) and inst.upper().startswith('SWAP') and\
                            inst in self.instruments.keys():
                        self.load_data_row(
                            position=inst, rate=self.instruments[inst].r_swap,
                            date=self.instruments[inst].maturity,
                            typ=inst, origin=int(self.instruments[inst].is_market_quote.value))
                    elif not isinstance(inst, bool) and inst.upper().startswith('BOND') and\
                            inst in self.instruments.keys():
                        self.load_data_row(
                            position=inst, rate=self.instruments[inst].coupon.coupon,
                            date=self.instruments[inst].maturity,
                            typ="FIXEDCOUPONBOND", origin=1)
                    else:
                        if self.dbg:
                            print("Warning (instrument) %s %s not found" % (inst, dte))


                    if not isinstance(inst, bool) and self.dbg:
                        print(item, inst, str(bool(self.results.loc[inst, 'loaded'] > 0)))
                else:
                    print("Warning (date) %s not found" % (dte))

            if self.curve_method == 0 and 'interpolated_instruments' in self.options.keys():
                for key, item in self.options['interpolated_instruments'].items():
                    self.interpolate_instruments(key, item)
        else:
            raise ValueError("Missing class elements!!!")

    def load_cf_matrices(self):
        ''' loads cash matrix elements (self.cf_matrix '''
        if self.cf_matrix is not None and all(self.cf_matrix.shape) > 0:
            for index in self.cf_matrix.index:
                dte = self.construct_date_str(index)
                if dte in self.cf_dates.keys():
                    inst = self.determine_instrument_name(dte)
                    if isinstance(inst, bool):
                        continue #  skip loop if date not an instrument

                    base = self.results.loc[inst]
                    if intbase.rate_instruments.LIBOR == int(base.type):
                        self.cf_matrix.loc[dte][inst] = (1 + 0.01*base['maturity']*base['rate'])

                        self.cf_prices[inst] = 1.

                    elif intbase.rate_instruments.FUTURE == int(base.type):
                        for loc, val in self.instruments[inst].cash_flow_df.iterrows():
                            self.cf_matrix.loc[loc, inst] = val['CF']

                    elif intbase.rate_instruments.SWAP == int(base.type):
                        self.cf_prices[inst] = self.instruments[inst].cash_flow_df.loc[
                            self.instruments[inst].reset.to_date(), 'price']

                        if  self.instruments[inst].t0_equal_T0:
                            for loc, val in self.instruments[inst].cash_flow_df.iterrows():
                                if self.instruments[inst].reset < loc:
                                    self.cf_matrix.loc[loc, inst] = val['CF']
                        else:
                            for loc, val in self.instruments[inst].cash_flow_df.iterrows():
                                self.cf_matrix.loc[loc, inst] = val['CF']
                    elif intbase.rate_instruments.FIXED_RATE_BOND == int(base.type):
                        self.cf_prices[inst] = self.instruments[inst].price
                        for loc, val in self.instruments[inst].cash_flow_df.iterrows():
                            self.cf_matrix.loc[loc, inst] = val['CF']
                    else:
                        raise ValueError("Element Excluded!!! -- will impact zero calculations")

                    self.results.loc[inst]['loaded'] += 1
        else:
            raise ValueError("Missing key matrix elements")

    def calc_date_inclusion(self, key):
        ''' determines whether key should be included in CF matrix
        returns: include -- True include CF date in cf_matrix, exlcusion  occurs ONLY all
        instruments are SWAP and to_equal_T0 True
        '''
        include = True
        if key in self.cf_dates.keys():
            exclude = True
            for itm in self.cf_dates[key]['instruments']:
                if itm.upper().endswith("CF") or itm.upper().endswith("NOTIONAL"):
                    exclude = False #  implies at least one instrument has cash flow
                    break
                else:
                    itm_split = itm.split(self.options['control']['split'])
                    if itm_split[0] in self.instruments.keys() and\
                        self.instruments[itm_split[0]].instrument_type ==\
                            intbase.rate_instruments.SWAP:
                        exclude = (exclude and self.instruments[itm_split[0]].t0_equal_T0)
                    else:
                        exclude = False
                        break
            include = not exclude

        return include

    def load_data_row(self, position='LIBOR1', rate=0.15, date='2012-10-02',
                      typ='LIBOR', origin=None):
        ''' Loader -- operates based on typ '''

        if "origin" in self.options['control']["columns"] and origin is None:
            self.results.loc[position]['origin'] = intbase.load_types.MARKET.value
        elif "origin" in self.options['control']["columns"] and\
                isinstance(origin, (int, float)):
            self.results.loc[position]['origin'] = origin

        if isinstance(date, float):
            mat = date
        else:
            mat = intdate.calc_bdte_diff(date, self.options)
        # print(position, date, mat)

        if typ.upper().startswith('LIBOR'):
            self.results.loc[position]['maturity'] = mat
            self.results.loc[position]['rate'] = rate
            self.results.loc[position]['type'] = intbase.rate_instruments.LIBOR.value

        elif typ.upper().startswith("FUTUR"):
            self.results.loc[position]['maturity'] = mat
            self.results.loc[position]['rate'] = intbase.futures_rate(rate)
            self.results.loc[position]['type'] = intbase.rate_instruments.FUTURE.value
        elif typ.upper().startswith("SWAP"):
            self.results.loc[position]['maturity'] = mat
            self.results.loc[position]['rate'] = rate
            self.results.loc[position]['type'] = intbase.rate_instruments.SWAP.value
        elif typ.upper().startswith("FIXEDCOUPONBO") or typ.upper().startswith("ZERO"):
            self.results.loc[position]['maturity'] = mat
            self.results.loc[position]['rate'] = rate
            self.results.loc[position]['type'] = intbase.rate_instruments.ZERO_COUPON.value if\
                typ.upper().startswith("ZERO") else intbase.rate_instruments.FIXED_RATE_BOND.value
        else:
            raise ValueError("Type not supported")

        self.results.loc[position, 'loaded'] = 1

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

    def determine_reference_date(self, instrument):
        ''' deterimines reference date and maturity '''
        refereposition = None
        maturity = np.nan

        if instrument in self.options['instruments'].keys():
            reference_date = self.options['instruments'][instrument]['ref_date']
        elif int(self.results.loc[instrument]['origin']) == 2:
            x0_nme = self.options['interpolated_instruments'][instrument]['reference_positions'][0]
            x1_nme = self.options['interpolated_instruments'][instrument]['reference_positions'][1]

            if self.options['instruments'][x0_nme]['date'] <=\
                    self.options['instruments'][x1_nme]['date']:
                x1_nme = x0_nme

            reference_date = self.options['instruments'][x1_nme]['ref_date']
        else:
            raise ValueError("Faulty FUTURES")

        refereposition = self.determine_instrument_name(reference_date)
        # print(refereposition, inst, "\n")
        # print(self.results.loc[inst], self.results.loc[refereposition])
        if not isinstance(refereposition, bool) and refereposition in self.results.index:
            maturity = self.results.loc[refereposition, 'maturity']
        else:
            maturity = intdate.calc_bdte_diff(reference_date, self.options)

        return reference_date, maturity

    def construct_date_str(self, val):
        ''' Constructs date as string from object '''
        day = (str(val.day) if val.day > 9 else ("0" + str(val.day)))
        month = (str(val.month) if val.month > 9 else ("0" + str(val.month)))
        return self.options['control']['split'].join([str(val.year), month, day])


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

            mat = intdate.calc_bdte_diff(item_dict['date'], self.options)

            dist = self.results.loc[position2]['maturity'] - self.results.loc[position1]['maturity']
            q = (mat - self.results.loc[position1]['maturity']) / dist

            rate = (1- q)*self.results.loc[position1]['rate'] +\
                    q*self.results.loc[position2]['rate']

            if result_position.upper().startswith('LIBOR'):
                self.load_data_row(position=result_position, rate=rate, date=item_dict['date'],
                                   typ='LIBOR', origin=intbase.load_types.INTERPOLATED.value)
            else:
                if position1 in self.options['instruments'].keys():
                    if typ.upper().startswith("FUTUR"):
                        self.instruments[result_position] = intrate.interest_rate_future(
                            result_position, rate, maturity=item_dict['date'],
                            reset=self.instruments[position1].reset,
                            frequency=self.instruments[position1].frequency,
                            options=self.options, dbg=False)


                        self.load_data_row(
                            position=result_position,
                            rate=self.instruments[result_position].rate,
                            date=self.instruments[result_position].maturity,
                            typ='FUTURE', origin=intbase.load_types.INTERPOLATED.value)
                    else:
                        self.load_data_row(position=result_position, rate=rate,
                                           date=item_dict['date'],
                                           typ=typ, origin=intbase.load_types.INTERPOLATED.value)
                else:
                    print("Warning no item %s loaded" % (result_position))

            if self.dbg:
                print("Interpolate Results: %f %f  %d  %d" % (
                    self.results.loc[result_position]['maturity'],
                    self.results.loc[result_position]['rate'],
                    int(self.results.loc[result_position]['type']),
                    int(self.results.loc[result_position]['origin'])))

        else:
            raise ValueError("(interpolate_instruments): Instrument Types MUST MATCH")

    def calc_exact_method0(self):
        ''' implements eaxct method with complete NxN matrix '''
        res = False
        if isinstance(self.cf_matrix, pd.DataFrame) and all(self.cf_matrix.shape) > 0 and\
                isinstance(self.cf_prices, pd.Series) and self.cf_prices.size > 0:
            dates, _ = intdate.calc_schedule_list(self.cf_dates.keys(), self.options)


            zeros = sci.linalg.solve(self.cf_matrix.transpose(), self.cf_prices)
            strt_dict = intdate.generate_schedule_dict(
                start=self.options['start_date'], period='S', count=dates,
                convention=self.options['control']['convention'],
                date_adjust=self.options['control']['date_adjust'])

            self.zeros = intdisc.discount_calculator(zeros, data_type=0, dates=strt_dict,
                                                     dbg=self.dbg)

            self.apply_zeros()
            res = True
        else:
            if self.dbg:
                print("Warning -- zeros NOT calculated")
        return res

    def calc_m_inverse(self):
        ''' calculates M inverse '''
        dim = self.cf_matrix.shape[0]
        M_inverse = np.ones([dim, dim])
        for j in np.arange(0, dim):
            if (j+1) < dim:
                M_inverse[j][j+1:] = 0.0


        if self.dbg:
            print("M_inverse %f %f " % (len(M_inverse), M_inverse.size/len(M_inverse)))
        return M_inverse

    def calc_w_inverse(self):
        ''' calculates W inverse '''
        dates = []
        for key, _ in self.cf_dates.items():
            if self.calc_date_inclusion(key):
                dates.append(key)

        _, vals = intdate.calc_schedule_list(dates, self.options)
        W_inverse = np.diag(np.sqrt(vals.transpose()[1]))

        if self.dbg:
            print("W_inverse %f %f " % (len(W_inverse), W_inverse.size/len(W_inverse)))
            print(vals.transpose()[1].min(), vals.transpose()[1].max(),
                  vals.transpose()[1].mean())

        return W_inverse

    def calc_A_fnl(self):
        ''' calculates A_fnl A^T*(A*A^T)^(-1)
        returns tuple w/ Afnl, M-invervse, W-inverse
        '''

        M_inverse = self.calc_m_inverse()
        W_inverse = self.calc_w_inverse()
        A = (self.cf_matrix.transpose().dot(M_inverse)).dot(W_inverse)
        A_fnl = A.dot(A.transpose())
        A_fnl = np.linalg.inv(A_fnl)
        A_fnl = A.transpose().dot(A_fnl)

        if self.dbg:
            print("A %f %f " % (len(A), A.size/len(A)))
            print("A_fnl %f %f " % (len(A_fnl), A_fnl.size/len(A_fnl)))

        return (A_fnl, M_inverse, W_inverse)

    def calc_exact_method1(self):
        ''' implements exact weighting method kxN (k << N) '''
        res = False
        if isinstance(self.cf_matrix, pd.DataFrame) and all(self.cf_matrix.shape) > 0 and\
                isinstance(self.cf_prices, pd.Series) and self.cf_prices.size > 0:

            dim = self.cf_matrix.shape[0]
            mult = np.zeros(dim)
            mult[0] = 1.0
            result = self.calc_A_fnl()

            self.deltas = (self.cf_prices -  (self.cf_matrix.transpose().dot(result[1])).dot(mult))
            self.deltas = (result[0].dot(self.deltas.values))
            if self.dbg:
                print("cf_matrix %f %f " % (self.cf_matrix.shape[0], self.cf_matrix.shape[1]))
                print("deltas %f %f " % (len(self.deltas), self.deltas.size/len(self.deltas)))
                # print("mult %f %f " % (len(mult), mult.size/len(mult)))
                print("cf_prices (len) %d GT(0) %d %f " % (
                    len(self.cf_prices), (self.cf_prices > 0.0).sum(),
                    self.cf_prices.size/len(self.cf_prices)))

                # print("M_inverse Diag")
                # print(M_inverse.diagonal())
                # print("Items %f %f %f" % (M_inverse[0][dim-1], M_inverse[2][1], M_inverse[2][3]))
                # print("W_inverse Diag")
                # print(W_inverse.diagonal())
                # print("Items %f %f %f" % (W_inverse[0][dim-1], W_inverse[2][1], W_inverse[2][3]))
                # print("Deltas")
                # print(self.deltas)
                # print("A_fnl*A_fnl.inv")
                # print(B_fnl.dot(A_fnl).diagonal())

            zeros = result[1].dot((result[2].dot(self.deltas) + mult))

            dates, _ = intdate.calc_schedule_list(self.cf_dates.keys(), self.options)
            strt_dict = intdate.generate_schedule_dict(
                start=self.options['start_date'], period='S', count=dates,
                convention=self.options['control']['convention'],
                date_adjust=self.options['control']['date_adjust'])

            origin_dates = []
            for itm in self.instruments.values():
                origin_dates.append(itm.maturity)
            self.zeros = intdisc.discount_calculator(zeros, data_type=0, dates=strt_dict,
                                                     origin=origin_dates, dbg=self.dbg)

            self.apply_zeros()
            res = True
        else:
            if self.dbg:
                print("Warning -- zeros NOT calculated")
        return res

    def apply_zeros(self, mapping_dict=None, override=False):
        ''' Updates self.results with zeros, maturity, yields '''
        res = (True if override else self.zeros.status())

        if mapping_dict:
            m_dict = mapping_dict.copy()
        else:
            m_dict = {"zero": "zero", "maturity": "maturity", "date_diff": "date_diff",
                      "forward": "forward", "yield": "yield"}

        if res:
            for loc in self.results.index:
                indx = self.zeros.determine_closest_maturity(self.results.loc[loc, 'maturity'])
                if indx:
                    self.results.loc[loc, 'zero'] = self.zeros.matrix.loc[indx, m_dict['zero']]
                    self.results.loc[loc, 'maturity'] = self.zeros.matrix.loc[indx,
                                                                              m_dict['maturity']]
                    self.results.loc[loc, 'date_diff'] = self.zeros.matrix.loc[indx,
                                                                               m_dict['date_diff']]
                    self.results.loc[loc, 'forward'] = self.zeros.matrix.loc[indx,
                                                                             m_dict['forward']]
                    self.results.loc[loc, 'yield'] = self.zeros.matrix.loc[indx, m_dict['yield']]
                else:
                    if self.dbg:
                        print("Maturity (%f) not found" % (self.results.loc[loc, 'maturity']))
        else:
            if self.dbg:
                print(self.zeros)

    def apply_yield_forward_calcs(self):
        ''' Calculates spot, yield and forwards based on provides zeros
        '''
        for indx, vals in self.results.iterrows():
            if vals.type == 2. or vals.type == 4.:
                date_diff = intdate.convert_period_to_year(self.instruments[indx].frequency,
                                                           self.options)
                if abs(date_diff - vals['date_diff']) > 0.025:
                    strt = intdate.convert_date_bdte(self.options['start_date'], self.options)
                    maturity = strt.get_day_count(self.instruments[indx].reset,
                                                  self.options['control']['convention'])
                    ref_indx = self.zeros.determine_closest_maturity(maturity)
                    if ref_indx:
                        if self.dbg:
                            print("Warning Updating forward rate %s %f" % (
                                indx, abs(date_diff-vals['date_diff'])))
                        frwrd = intbase.calc_forward_rate_1d(
                            self.zeros.matrix.loc[ref_indx, 'maturity'],
                            self.results.loc[indx, 'maturity'],
                            self.zeros.matrix.loc[ref_indx, 'zero'],
                            self.results.loc[indx, 'zero'], 100.)

                        self.results.loc[indx, 'forward'] = frwrd
                        ref_indx = self.zeros.determine_closest_maturity(
                            self.results.loc[indx, 'maturity'])

                        if ref_indx:  # Updates zeros forward
                            self.zeros.matrix.loc[ref_indx, 'forward'] = frwrd
            self.results.loc[indx, 'spot'] = intbase.calc_spot_simple_1d(
                vals['zero'], vals['maturity'], 100.)
