''' Common pricing methods corresponding to Interest rate Instruments '''
import datetime as dt
#from collections import OrderedDict
import json
import os
import scipy as sci
import numpy as np
import pandas as pd
# import interest_rate_base as intbase
import interest_rate_dates as intdate
import interest_rate_discount_lorimier as intdisc
import curve_constructor as cb
import pyfi_filter as fi_filter


class curve_builder_lorimier(cb.curve_builder):
    ''' Class constructing applying LORIMEIR mmethod:
    methods: (3): Hilbert Space Smoothing Spline
    '''

    def __init__(self, options, alpha=0, dbg=False):
        ''' Constructor Depends options dictionary && dataframe of
            -  zero coupon yields and maturities

        '''
        if isinstance(options, str) and os.path.exists(options):
            with open(options, "r") as fp:
                init_options = json.load(fp)
            fp.close()
        elif isinstance(options, dict):
            init_options = options.copy()
        else:
            raise ValueError("Faulty -- options specification ")

        self.df = None
        self.alpha = alpha
        self.determine_instruments(init_options, init_options['data']['file'], dbg=dbg)
        super().__init__(init_options, method=3, dbg=dbg)
        self.calc_exact_method0()

    def determine_instruments(self, options, filename, dbg=False):
        ''' Loads Data Frame of acceptable instruments '''
        if 'instruments' not in options.keys() and isinstance(filename, str) and\
                os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
        elif 'instruments' in options.keys():
            if dbg:
                print("instruments already exist")
        else:
            raise ValueError("No Instruments Found")

        if isinstance(df, pd.DataFrame) and all(df.shape) > 0:
            self.df = df.copy()
            if self.df.shape[0] < 1:
                raise ValueError("Data Frame contains only excluded items")

            # TODO Apply repeated filter concept here
            if 'filter' in options.keys() and isinstance(options['filter'], dict):
                obj = self.determine_filter_obj(options['filter'], dbg)

                ind_inc = obj.include()
                self.df = self.df[ind_inc]
                self.df['excluded'] = False
            elif 'filter' in options.keys() and isinstance(options['filter'], list):
                for itm in options['filter']:
                    obj = self.determine_filter_obj(itm, dbg)

                    ind_inc = obj.include()
                    self.df = self.df[ind_inc]
                    self.df['excluded'] = False
        else:
            raise ValueError("Empty Data Frame")

        options["instruments"] = {}

        for cnt, itm in enumerate(self.df.iterrows()):
            name = ("PRINC_STRIP" if itm[1]['Description'].find("STRIPPED PRIN") >= 0 else
                    "TREAS_BILL")

            day_count = ("30_360" if itm[1]['Description'].find("STRIPPED PRIN") >= 0 else
                         "act_360")
            name = name + str(cnt+1)
            if "Price" in self.df.columns:
                price = itm[1]['Price']

            if "maturity_date" in self.df.columns:
                maturity_date = dt.datetime.strptime(itm[1]['maturity_date'], "%m/%d/%Y")
                maturity_date = dt.datetime.strftime(maturity_date,
                                                     options["control"]["date_format"])

            options["instruments"][name] = {"type": "ZERO COUPON", "price": price,
                                            "date": maturity_date, "day_count": day_count}
    def determine_filter_obj(self, item, dbg=False):
        ''' Calculates Filter object '''
        if isinstance(item, dict) and "operator" in item.keys():
            obj = fi_filter.fixed_income_filter_logical(item, self.df, dbg=dbg)
        elif isinstance(item, dict) and "type" in item.keys() and\
                item['type'].upper() == "NUMERIC":
            obj = fi_filter.fixed_income_filter_numeric(item, self.df, dbg=dbg)
        elif isinstance(item, dict) and "type" in item.keys() and\
                item['type'].upper() == "TEXT":
            obj = fi_filter.fixed_income_filter(item, self.df, dbg=dbg)
        elif isinstance(item, dict) and "type" in item.keys() and\
                item['type'].upper() == "REPEATED_MATURITY":
            obj = fi_filter.repeated_maturity_filter(item, self.df, dbg=dbg)
            if dbg:
                print("Repeated Count %d" % (obj.get_gt_one_count()))
        else:
            if dbg:
                print(item)
            raise ValueError("Faulty Item")
        return obj


    def build_arrays(self):
        ''' Constructs DataFrame elements '''
        instrument_cnt = len(self.instruments)
        if self.cf_matrix is None:
            self.cf_matrix = np.zeros([instrument_cnt+1, instrument_cnt+1])
            self.cf_prices = np.zeros([instrument_cnt+1])

            for i, zero in enumerate(self.instruments.values()):
                self.cf_prices[i+1] = self.alpha*zero.get_maturity()*zero.get_yield()
                self.cf_matrix[0][i+1] = zero.get_maturity()
                self.cf_matrix[i+1][0] = self.alpha*zero.get_maturity()

            for i, zero_i in enumerate(self.instruments.keys()):
                for j, zero_j in enumerate(self.instruments.keys()):
                    self.cf_matrix[i+1][j+1] = self.lorimier_dot_prod(zero_i, zero_j)

        if self.results is None: # # instruments X column count
            if "columns" in self.options['control'].keys():
                rows = len(self.options['control']["columns"])
                mtrx = np.zeros([instrument_cnt, rows])
                names_sorted = sorted(self.names, key=lambda x: self.names[x])
                self.results = pd.DataFrame(mtrx, index=names_sorted,
                                            columns=self.options['control']['columns'])

            if self.dbg:
                print("Construction NP array shape %i length %i" % (self.results.shape[0],
                                                                    self.results.shape[1]))

    def load_cf_results(self):
        ''' Loads results elemt upto (but not including) zeros '''
        for key, zero in self.instruments.items():
            self.load_data_row(position=key, rate=zero.get_price(),
                               date=zero.get_maturity(),
                               typ='ZERO')

    def load_cf_matrices(self):
        ''' loads cash matrix elements (self.cf_matrix '''

    def calc_exact_method0(self):
        ''' implements eaxct method with complete NxN matrix '''
        res = False
        if isinstance(self.cf_matrix, np.ndarray) and all(self.cf_matrix.shape) > 0 and\
                isinstance(self.cf_prices, np.ndarray) and\
                self.cf_prices.shape[0] == self.cf_matrix.shape[0] == self.cf_matrix.shape[1]:
            dates_dict = self.options.copy()
            dates_dict['parameters'] = {}
            dates_dict['parameters']['alpha'] = self.alpha

            dates, _ = intdate.calc_schedule_list(
                self.cf_dates.keys(), self.options)

            beta = sci.linalg.solve(self.cf_matrix, self.cf_prices)
            if self.dbg:
                print(beta)
            dates_dict['parameters']['beta'] = beta.copy()
            dates_dict['parameters']['tau'] = []
            for itm in self.instruments.values():
                dates_dict['parameters']['tau'].append(itm.get_maturity())

            dates_dict['parameters']['dates'] = intdate.generate_schedule_dict(
                start=self.options['start_date'], period='S', count=dates,
                convention=self.options['control']['convention'],
                date_adjust=self.options['control']['date_adjust'])

            prices = [bnd.get_price() for bnd in self.instruments.values()
                      if isinstance(bnd, cb.intrate.fi_instrument)]

            self.zeros = intdisc.discount_calculator_lorimier(
                prices, data_type=3, dates=dates_dict, dbg=self.dbg)

            mapping_dict = {"zero": "zero", "maturity": "maturity", "date_diff": "date_diff",
                            "forward": "forward", "yield": "yield_hat"}

            override = bool("control" in self.options.keys() and "override" in
                            self.options['control'].keys() and
                            int(self.options['control']['override']) > 0)

            self.apply_zeros(mapping_dict, override)
            res = True
        else:
            if self.dbg:
                print("Warning -- zeros NOT calculated")
        return res

    def lorimier_dot_prod(self, h_i, h_j):
        ''' Calculates dot product of h_i, h_j  based on lorimeir definition'''
        v1 = self.instruments[h_i].get_maturity()
        v2 = self.instruments[h_j].get_maturity()

        res = v1*v2 + 0.5*min(v1, v2)**2*max(v1, v2) - min(v1, v2)**3/6.
        res = (self.alpha*res + (1.0 if v1 == v2 else 0.0))
        if self.dbg:
            print(h_i, h_j, v1, v2)

        return res
