''' testing routines correcpsonding to interest_rate_base.py '''
import json
import pytest
import pandas as pd
import businessdate as bdte
import interest_rate_swap_convenience as swapconv
import interest_rate_instruments as int_rate


@pytest.fixture
def tolerance():
    return 5.0e-4

class loader():
    ''' loader -- setup for testing '''
    def __init__(self, rates_file='./data/caliibrated_base.csv',
                 json_file='./data/wk5_psi_fitting_v2.json', dbg=False):
        with open(json_file, "r") as fp:
            self.options = json.load(fp)
        fp.close()

        self.cap_dict = {}
        self.swap_dict = {}

        self.dbg = dbg
        self.rates = pd.read_csv(rates_file, sep=",", index_col=0, header=0, parse_dates=True)
        ndx = [bdte.BusinessDate(year=itm.year, month=itm.month, day=itm.day)
               for itm in self.rates.index]

        self.rates.index = sorted(ndx)

        self.mkt_vals = self.options['cap_prices'].copy()
        swap_params_dict = {'type': 'SWAP', 'date': '', 'rate': 0.0, 'reset_date': '2013-04-01',
                            'frequency': 'S', 'reference_rate': 'LIBOR'}

        for nme, val in self.mkt_vals.items():
            # dbg = bool(nme == 'CAP1' or nme == 'CAP5')
            nm1 = nme.replace('C', 'SW')
            swap_params_dict['date'] = val['date']
            swap_params_dict['rate'] = val['strike']

            self.swap_dict[nm1] = swapconv.build_swap(nm1, swap_params_dict,
                                                      self.options, dbg=False)

            if self.dbg:
                print("Caclulating strikes")
            #self.mkt_vals[nme]['strike'] = 100.*self.swap_dict[nm1].calc_swap_strike_zeros(
            #    self.rates)

            if self.dbg:
                print("Running %s strike %f" % (nme, self.mkt_vals[nme]['strike']))

            dbg = False # (nme.upper() == 'CAP1' or nme.upper() == 'CAP5')

            self.cap_dict[nme] = int_rate.cap(nme, val['strike'], val['date'],
                                              swap_params_dict['reset_date'],
                                              self.options, frequency=swap_params_dict['frequency'],
                                              dbg=dbg)

            if self.dbg:
                print("No vega found for %s" % (nme))
            sigma_fnl = 0.01*val['volatility']
            self.mkt_vals[nme]['vega'] = self.cap_dict[nme].vega_cap(self.rates,
                                                                     sigma_fnl, hjm_model=None)


@pytest.fixture
def strt():
    return loader()

def test_cap_solve_1(strt, tolerance):
    ''' test cap solver ealuvator 1 '''
    x1 = strt.cap_dict['CAP1'].price_cap_solver(0.01*strt.mkt_vals['CAP1']['volatility'],
                                                strt.rates, strt.mkt_vals['CAP1']['price'],
                                                dbg=True)
    assert abs(x1) < tolerance


def test_cap_solve_2(strt, tolerance):
    ''' test cap solver ealuvator 2 '''
    x1 = strt.cap_dict['CAP2'].price_cap_solver(0.01*strt.mkt_vals['CAP2']['volatility'],
                                                strt.rates, strt.mkt_vals['CAP2']['price'],
                                                dbg=True)
    assert abs(x1) < tolerance


def test_cap_solve_5(strt, tolerance):
    ''' test cap solver ealuvator 5 '''
    x1 = strt.cap_dict['CAP5'].price_cap_solver(0.01*strt.mkt_vals['CAP5']['volatility'],
                                                strt.rates, strt.mkt_vals['CAP5']['price'],
                                                dbg=True)
    assert abs(x1) < tolerance

def test_swap_1(strt, tolerance):
    ''' strike calculator swap 1 '''
    x1 = strt.swap_dict['SWAP1'].calc_swap_strike_zeros(strt.rates)

    assert abs(x1 - 0.0054) < tolerance


def test_swap_5(strt, tolerance):
    ''' strike calculator swap 5 '''
    x1 = strt.swap_dict['SWAP5'].calc_swap_strike_zeros(strt.rates)

    assert abs(x1 - 0.0148) < tolerance

def test_implied_vol_black_1(strt, tolerance):
    ''' tests calculation of implied volitility for cap 1 --  black '''
    x1 = strt.cap_dict['CAP1'].calc_implied_volatility(strt.rates, price=0.00122693, dbg=True)
    assert abs(x1 - 1.7052) < tolerance

def test_implied_vol_black_5(strt, tolerance):
    ''' tests calculation of implied volitility for cap 5 --  black '''
    x1 = strt.cap_dict['CAP5'].calc_implied_volatility(strt.rates, price=0.0210, dbg=True)
    assert abs(x1 - 0.4136) < tolerance

def test_implied_vol_bachelier_1(strt, tolerance):
    ''' tests calculation of implied volitility for cap 1 --  bachelier '''
    x1 = strt.cap_dict['CAP1'].calc_implied_volatility(
        strt.rates, price=0.0012, hjm_model='bachelier', dbg=True)
    assert abs(x1 - 0.008681) < tolerance

def test_implied_vol_bachelier_5(strt, tolerance):
    ''' tests calculation of implied volitility for cap 5 --  bachelier '''
    x1 = strt.cap_dict['CAP5'].calc_implied_volatility(
        strt.rates, price=0.0210, hjm_model='bachelier', dbg=True)
    assert abs(x1 - 0.006386) < tolerance
