''' Class for evaliuating quality of discount class '''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import interest_rate_discount as intdisc
# import curve_builder_lorimier as cvl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'


class discount_calculator_review():
    '''Class for evaliuating quality of discount class'''
    def __init__(self, options, dc, dbg=True):
        ''' instantiates review tool '''
        if isinstance(dc, intdisc.discount_calculator):
            self.dc_ = dc
        else:
            raise ValueError("dc must be of type discount_calculator")

        if options and isinstance(options, dict):
            self.options = options.copy()
        else:
            raise ValueError("options must be of type dictionary")

        self.dbg = dbg
        self.results = None
        self.results_stats = {}


    def review(self):
        ''' Constructs dictionary of comparisons based on the init_dict specification '''
        results_dict = {}
        includes = (self.dc_.matrix['maturity'] > 0.0)

        for key, val in self.options['comparisons'].items():
            name = (val['name'] if isinstance(val, dict) and 'name' in val.keys() else
                    '_'.join([key, "diff"]))

            results_dict[name] = np.abs(
                self.dc_.matrix.loc[includes, key] - self.dc_.matrix.loc[includes, val['comp']])

            self.results_stats[name] = results_dict[name].describe()
            if self.dbg:
                print(key)
                print(self.results_stats[name])

        self.results = pd.DataFrame(results_dict)

    def plot_act_projected(self, actual, f0=None, title=None):
        ''' plots actual versus projected '''
        # TODO: start here -- add init_dict and hard coded values below to json
        points = (int(self.options["plots"]["actual_v_proj"][actual])
                  if "plots" in self.options.keys() and "actual_v_proj" in
                  self.options["plots"].keys() and actual in
                  self.options["plots"]["actual_v_proj"].keys() and
                  isinstance(self.options["plots"]["actual_v_proj"][actual], (int, float))
                  else 85)

        x_fnl = np.zeros([2, points])
        includes = (self.dc_.matrix['maturity'] > 0)
        x_min = np.floor(np.min(self.dc_.matrix.loc[includes, 'maturity']))
        x_max = np.ceil(np.max(self.dc_.matrix['maturity']))

        x_fnl[0] = np.linspace(x_min, x_max, points)
        for loc, val in enumerate(x_fnl[0]):
            if f0:
                x_fnl[1][loc] = f0(val, False)
            else:
                x_fnl[1][loc] = self.dc_.f0(val, False)

        plt.figure(figsize=(10, 6))
        plt.plot(x_fnl[0], x_fnl[1], lw=1.5, label='1st')
        plt.plot(self.dc_.matrix.loc[includes, 'maturity'],
                 self.dc_.matrix.loc[includes, actual], 'ro', label='point')

        if title:
            plt.title(title)
