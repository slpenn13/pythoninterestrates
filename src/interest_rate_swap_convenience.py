''' SWAP convenience functions '''
import json
import numpy as np
import scipy as sci
import interest_rate_base as intbase
import interest_rate_capfloor_convenience as capconv
import interest_rate_dates as intdate
import interest_rate_discount as intdisc
import interest_rate_instruments as intrate
# import interest_rate_dates as intdate
# import interest_rate_hjm as inthjm

def calc_price_black_swap_period(r_swap, strike, t0, sigma,
                                 is_payer=True, forward_t=0.0, dbg=False):
    ''' calculates black swap period
    r_swap: swap R_swap (numerator) in log calculation
    strike: swaption strike
    t0: swaption strike date
    zero_t2: zero coupon bopnd price @ time t2
    is_payer: bool indicating whether payer receiver for swaption
    forward_t: time settled forward, defaults to 0.0
    dbg: indicates whether to output debugging output
    '''

    d1, d2 = capconv.calc_di_black(strike, t0, r_swap, sigma, forward_t=forward_t, dbg=dbg)
    if is_payer:
        p1 = r_swap*capconv.norm_dist.cdf(d1)
        p2 = strike*capconv.norm_dist.cdf(d2)
        if dbg:
            print("swaption  (black--payer) r_swap %f strike %f" % (
                r_swap, strike))
            print("swaption (black--payer) p1 %f p2 %f swap_period %f" % (p1, p2, (p1-p2)))
    else:
        p2 = r_swap*(1. - capconv.norm_dist.cdf(d1))
        p1 = strike*(1. - capconv.norm_dist.cdf(d2))

        if dbg:
            print("swaption (black) r_swap %f strike %f" % (
                r_swap, strike))
            print("swaption (black) p1 %f p2 %f swap_period %f" % (p1, p2, (p1-p2)))

    swp = (p1 - p2)

    return swp

def calc_price_bachelier_swap_period(r_swap, strike, t0, sigma, is_payer=True, forward_t=0.0,
                                     dbg=False):
    ''' calculates bachelier swap period
    r_swap: swap R_swap (numerator) in log calculation
    strike: swaption strike
    t0: swaption strike date
    zero_t2: zero coupon bopnd price @ time t2
    is_payer: bool indicating whether payer receiver for swaption
    forward_t: time settled forward, defaults to 0.0
    dbg: indicates whether to output debugging output
    '''
    mult = sigma*np.sqrt(t0 - forward_t)

    d1 = capconv.calc_di_bachelier(strike, t0, r_swap, sigma, forward_t=forward_t, dbg=dbg)
    if is_payer:
        p1 = d1*capconv.norm_dist.cdf(d1)
        p2 = capconv.norm_dist.pdf(d1)
        if dbg:
            print("swaption  (bachelier) r_swap %f mult %f" % (r_swap, mult))
            print("swaption (bachelier) p1 %f p2 %f swap_period %f" % (p1, p2, mult*(p1+p2)))
    else:
        p1 = (-d1)*capconv.norm_dist.cdf(-d1)
        p2 = capconv.norm_dist.pdf(-d1)

        if dbg:
            print("swaption (bachelier) r_swap %f mult %f" % (r_swap, mult))
            print("swaption (bachelier) p1 %f p2 %f swap_period %f" % (p1, p2, mult*(p1+p2)))

    swp = mult*(p1 + p2)

    return swp

def generate_swap_dictionary(options, maturity, swap_start=None, swapid=0, reference="SWAP1",
                             rates=None, princ=1.0, frequency='Y', reset_date=None,
                             to_equal_T0=False, dbg=False):
    ''' constructs dictionary of swap assumptions necessary to construction of swap instrument
    options: python dict including control and instruments dictionaries
    maturity: maturity of constructed swap
    swap_start: aplies in case interpolating between two swap rates
    swapid: string applied to identify SWAP
    reference: refers to swap of same name in options['instruments']
    rates: asscepts (1) single float or (2) 2x2 conformable matrix object
    NOTE: items procede by asterisk are applied only in case not reference object provided
    princ (*): principal
    frquency (*): frequency of swap payments
    reset_date (*) date (date conformable object) of reset
    to_equal_T0 (*) bool indicating whether valuation time equals reset date
    dbg: debugging indicator
    returns: (1) swap specfication dict (2) swap_name
    '''
    if reference and isinstance(reference, str):
        if 'instruments' in options.keys() and reference in options['instruments'].keys():
            new_swap_dict = options['instruments'][reference].copy()
        else:
            raise ValueError("genrate_swap_dictionary: missing instruments + reference")
    else:
        if dbg:
            print("Warning -- applying default SWAP definition ")
        new_swap_dict = {'type': 'SWAP', 'princ': princ, 'frequency': frequency,
                         'reference_rate': 'LIBOR', 'to_equal_T0': to_equal_T0}
        new_swap_dict['reset_date'] = reset_date

    new_swap_dict['date'] = maturity

    if rates and isinstance(rates, float) and np.isfinite(rates):
        new_swap_dict['rate'] = rates
        new_swap_dict['is_market'] = 1
    elif rates and  isinstance(rates, (list, np.ndarray)) and np.size(rates) > 3:
        new_swap_dict['is_market'] = 0
        if isinstance(rates, list) and len(rates) == 2:
            np_rates = np.array(rates, ndmin=2)
        elif isinstance(rates, np.ndarray):
            np_rates = rates.copy()
        else:
            raise ValueError("genrate_swap_dictionary: faulty rates spec")

        date_diff_dbl2 = intdate.calc_bdte_diff(maturity, options, swap_start)
        new_swap_dict['rate'] = sci.interp(
            date_diff_dbl2, xp=np_rates[0], fp=np_rates[1])
    else:
        new_swap_dict['rate'] = np.nan
        new_swap_dict['is_market'] = 0
        print("Warning (genrate_swap_dictionary): faulty rates specifcation")

    if swapid > 0:
        new_swap_name = "".join(["SWAP", str(swapid)])
    else:
        new_swap_name = "SWAP"

    return new_swap_dict, new_swap_name

def build_swap(name, swap_dict, options, dbg=False):
    ''' Helper function -- Constructs SWAP from dictionary '''
    if swap_dict["type"].upper() == 'SWAP':
        princ = (swap_dict['princ'] if 'princ' in swap_dict.keys() else 1.0)
        cpn_fixed = intbase.fixed_coupon(coupon=swap_dict['rate'],
                                         frequency=swap_dict['frequency'])

        lg1 = intrate.fixed_coupon_bond('FIXED', swap_dict['reset_date'], swap_dict['date'],
                                        options, cpn_fixed, princ, dbg=dbg)

        cpn_float = intbase.floating_coupon(reference_rate=swap_dict["reference_rate"],
                                            frequency=swap_dict['frequency'])

        lg2 = intrate.floating_rate_bond("FLOATING", swap_dict['reset_date'], swap_dict['date'],
                                         options, cpn_float, princ, dbg)

        if 'is_market' in swap_dict.keys():
            is_market = bool(int(swap_dict['is_market']) > 0)
        else:
            is_market = True

        if "t0_equal_T0" in swap_dict.keys():
            t0_equal_T0 = bool(int(swap_dict["t0_equal_T0"]) > 0)
        else:
            t0_equal_T0 = None

        swap_final = intrate.swap(name, lg1, lg2, options, is_market,
                                  t0_equal_T0=t0_equal_T0,
                                  reset=swap_dict['reset_date'], dbg=dbg)

    else:
        raise ValueError("Dict type muyst be swap")

    return swap_final

def calc_swap_strike(reset, maturity, zeros, name='zero', dbg=False):
    ''' calculates swap strike based on DataFrame with column zero and index
    elements reset maturity
    '''
    num = (zeros.loc[reset, name] -  zeros.loc[maturity, name])
    x4_ind = [True if reset < itm <= maturity else False for itm in zeros.index]
    denom = (zeros.loc[x4_ind, 'zero'].dot(zeros.loc[x4_ind, 'date_diff']))
    strike = num / denom

    if dbg:
        print("calc_swap_strike: num %f denom %f strike %f" % (
            num, denom, strike))


    return strike

class swaption():
    ''' swaption -- '''
    instrument_type = intbase.rate_instruments.SWAPTION

    def __init__(self, name, strike, swap, options, dbg=False):
        self.options = options
        self.debug = dbg
        if isinstance(swap, intrate.swap):
            self.reference_instrument = swap
            self.maturity = swap.reset
        elif isinstance(swap, json):
            self.maturity = intdate.convert_date_bdte(swap['reset_date'], options)
            self.reference_instrument = build_swap("SWAP", swap, options,
                                                   dbg=self.debug)
        else:
            raise ValueError("Must be type JSON or swap")
        self.maturity_dbl = intdate.calc_bdte_diff(self.maturity, self.options)
        self.strike = strike
        self.name = name
        self.notional = self.reference_instrument.princ

    def price_swaption(self, forward_t=0.0, sigma=None, zeros=None, mdl=None):
        ''' prices swaption for interest rate path '''
        price = np.NaN

        if sigma and zeros is not None and isinstance(zeros, intdisc.discount_calculator):
            print("now")

        elif sigma and zeros is not None and isinstance(zeros, intdisc.pd.DataFrame) and\
                all(zeros.shape) > 0:
            ind = (zeros.index > self.maturity)
            mult = self.notional*zeros.loc[ind, 'zero'].dot(zeros.loc[ind, 'date_diff'])

            # TODO -- (2020-05-24) start here -- TESTING
            if isinstance(mdl, str) and mdl.lower().startswith('bache'):
                price = calc_price_bachelier_swap_period(
                    0.01*self.reference_instrument.r_swap, 0.01*self.strike,
                    self.maturity_dbl,
                    sigma, is_payer=self.reference_instrument.is_fixed_payer,
                    forward_t=forward_t, dbg=self.debug)

            elif isinstance(mdl, str) and mdl.lower().startswith('black'):
                price = calc_price_black_swap_period(
                    0.01*self.reference_instrument.r_swap, 0.01*self.strike,
                    self.maturity_dbl, sigma,
                    is_payer=self.reference_instrument.is_fixed_payer,
                    forward_t=forward_t, dbg=self.debug)
            else:
                raise ValueError("swaption faulty -- mdl type")

            price *= mult
        else:
            raise ValueError("Faulty zeros / sigma")
        return price

    def price_swaption_solver(self, sigma, forward_t, zeros, price=0.0, mdl=None, dbg=False):
        ''' swaption price calculator '''
        dbg = (dbg or self.debug)
        result = self.price_swaption(forward_t=forward_t, sigma=sigma, zeros=zeros, mdl=mdl)
        if dbg:
            print("sigma %.8f target %f Value %.8f Diff %.8f" % (
                sigma, price, result, (price - result)))

        return price - result

    def calc_swaption_implied_volatility(self, forward_t=0.0, zeros=None, price=0.0,
                                         left=0.0005, right=2.0, mdl=None, dbg=False):
        ''' calculates implied volatility of interest rate path '''

        dbg = (dbg or self.debug)

        xresult = sci.optimize.brentq(self.price_swaption_solver, left, right, args=(
            forward_t, zeros, price, mdl, dbg), full_output=True)

        if dbg:
            print(xresult)

        return xresult[0]
