''' Cap Floor convenience functions '''
import numpy as np
from scipy.stats import norm as norm_dist
import interest_rate_base as intbase

def calc_di_black(strike, t1, forward, sigma, forward_t=0.0, dbg=False):
    ''' calculates black black d_i '''
    diff_t = (t1 - forward_t)
    d1 = (np.log(forward / strike) + 0.5*diff_t*sigma**2) / (sigma*np.sqrt(diff_t))
    d2 = (np.log(forward / strike) - 0.5*diff_t*sigma**2) / (sigma*np.sqrt(diff_t))

    if dbg:
        print("di (black) caplet: strike %f t1 %f forward_t %f forward %f sigma %f" %
              (strike, t1, forward_t, forward, sigma))
        print("di (black) caplet d1 %f d2 %f" % (d1, d2))

    return d1, d2

def calc_price_black_caplet(strike, t1, t2, zero_t1, zero_t2, sigma, dbg=False):
    ''' calculates black caplet zero '''
    delta = (t2 - t1)
    forward = intbase.calc_forward_rate_1d(t1, t2, zero_t1, zero_t2)
    mult = delta*zero_t2

    d1, d2 = calc_di_black(strike, t1, forward, sigma, dbg=dbg)
    p1 = mult*forward*norm_dist.cdf(d1)
    p2 = mult*strike*norm_dist.cdf(d2)
    if dbg:
        print(" caplet (black) delta %f forward %f mult %f" % (delta, forward, mult))
        print(" caplet (black) p1 %f p2 %f caplet %f" % (p1, p2, (p1-p2)))

    cpl = p1 - p2

    return cpl

def calc_vega_black_caplet(strike, t1, t2, zero_t1, zero_t2, sigma, dbg=False):
    ''' calculates black caplet vega '''

    forward = intbase.calc_forward_rate_1d(t1, t2, zero_t1, zero_t2)
    d1, _ = calc_di_black(strike, t1, forward, sigma, dbg=dbg)
    cpl_vega = (t2 - t1)*zero_t2*forward*np.sqrt(t1)*norm_dist.pdf(d1)

    if dbg:
        print("vega (black): strike %.7f forward %.7f t1 %f t2 %f " % (strike, forward, t1, t2))
        print("vega (black): p(0, t1) %.7f d1 %.7f caplet %.7f " % (zero_t1, d1, cpl_vega))

    return cpl_vega

def calc_di_bachelier(strike, t1, numerator, sigma, forward_t=0.0, dbg=False):
    ''' calculates bachelier di value '''
    diff_t = (t1 - forward_t)
    p1 = (numerator - strike)
    p2 = (sigma*np.sqrt(diff_t))

    if dbg:
        print("d_i (bachelier): forward %f strike %f sigma %f t1 %f" % (
            numerator, strike, sigma, t1))
        print("d_i (bachelier): p1 %f p2 %f result %f" %(p1, p2, p1/p2))


    return p1/p2

def calc_price_bachelier_caplet(strike, t1, t2, zero_t1, zero_t2, sigma, forward_t=0.0, dbg=False):
    ''' calculates caplet price based on bachelier (normal) method '''
    delta = (t2 -t1)
    numerator = intbase.calc_forward_rate_1d(t1, t2, zero_t1, zero_t2)

    d1 = calc_di_bachelier(strike, t1, numerator, sigma, forward_t=forward_t, dbg=dbg)
    mult = delta*zero_t2*sigma*np.sqrt(t1 - forward_t)
    p1 = mult*d1*norm_dist.cdf(d1)
    p2 = mult*norm_dist.pdf(d1)

    if dbg:
        print(" caplet (bachelier) delta %f zero %f mult %f" % (delta, zero_t2, mult))
        print(" caplet (bachelier) p1 %f p2 %f caplet %f" % (p1, p2, (p1+p2)))
    cpl = p1 + p2
    return cpl

def calc_vega_bachelier_caplet(strike, t1, t2, zero_t1, zero_t2, sigma, dbg=False):
    ''' calculates bachelier vega for caplet '''
    delta = (t2 -t1)
    numerator = intbase.calc_forward_rate_1d(t1, t2, zero_t1, zero_t2)
    d1 = calc_di_bachelier(strike, t1, numerator, sigma, dbg=dbg)

    p1 = delta*zero_t2*np.sqrt(t1)
    p2 = norm_dist.pdf(d1)

    if dbg:
        print("Vega (bachelier): zero %f t1 %f p1 %f p2 %f result %f" % (
            zero_t2, t1, p1, p2, p1*p2))

    return p1*p2


def calc_v_norm_1d(t=0, t0=0, t1=0, params=None, dbg=False):
    ''' calculates v norm  for sigmma(t, T) = exp^(kappa*(T - t))*sigma
    t: forward time < t0
    t0: first date to < t1
    t1: second date
    paramas: dictionary sigma and kappa must be params
    dbg: controls
    '''
    if params is None or 'sigma' not in params.keys():
        raise ValueError("No value for sigma provided")
    sigma = params['sigma']
    if params is None or 'kappa' not in params.keys():
        raise ValueError("No value for kappa provided")
    kappa = params['kappa']

    res1 = sigma**2/kappa**2*(np.exp(-kappa*t0) - np.exp(-kappa*t1))**2
    res2 = 0.5*(np.exp(2.*kappa*t0) - 1.)/kappa

    if dbg:
        print("kappa %f sigma %f t %f t0 %f t1 %f" %
              (kappa, sigma, t, t0, t1))
        print("itm1 %f itm2 %f" % (res1, res2))

    return res1*res2

def calc_v_norm_nd(t=0, t0=0, t1=0, params=None, dbg=False):
    ''' calculates v norm  for n dimensions '''
    if params is None or 'sigma' not in params.keys():
        raise ValueError("(calc_v_norm_nd) No value for sigma provided")
    sigma = params['sigma'].copy()

    if params is None or 'kappa' not in params.keys():
        raise ValueError("(calc_v_norm_nd) No value for kappa provided")

    kappa = params['kappa'].copy()

    if len(sigma) == len(kappa):
        cnt = min(len(sigma), len(kappa))
    else:
        raise ValueError("(calc_v_norm_nd) Sigma & kappa must match")

    result = 0.0

    for j in np.arange(0, cnt):
        res1 = (sigma[j]**2/kappa[j]**2)*(np.exp(-kappa[j]*t0) - np.exp(-kappa[j]*t1))**2
        res2 = 0.5*(np.exp(2.*kappa[j]*t0) - 1.)/kappa[j]

        if dbg:
            print("J(%d): kappa %f sigma %f t %f t0 %f t1 %f" % (
                (j, kappa[j], sigma[j], t, t0, t1)))

            print("J(%d): res1 %f res2 %f intermed %f" % (j, res1, res2, res1*res2))

        result += res1*res2

    return result
