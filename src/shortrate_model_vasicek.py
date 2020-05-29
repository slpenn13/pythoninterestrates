''' Implementation of vasicek class incorporating key calcultions '''
# import businessdate as bdte
import numpy as np
from scipy.stats import norm as norm_dist
# import pandas as pd
from interest_rate_hjm import hjm_model


class short_rate_vasicek(hjm_model):
    ''' implementation of Vasicek Short Rate Model '''
    def __init__(self, kappa, theta, sigma, r0, norm_method, dbg=False):
        ''' Vasicek implementation of short rate model '''
        super().__init__(kappa, theta, sigma, r0, calc_norm_method=norm_method)
        self.debug = dbg

        if self.debug:
            self.__repr__()

    def price_zero(self, t):
        ''' calculates zero coupon bond price for time t in Vasicek SRM '''
        const = -self.sigma**2/self.kappa**3 + 0.25*self.sigma**2/self.kappa**3 +\
                self.r0/self.kappa - self.theta/self.kappa
        calc_t = (self.theta + 0.5*self.sigma**2/self.kappa**2)*t
        calc_kt = (self.theta/self.kappa + self.sigma**2/self.kappa**3  -\
                self.r0/self.kappa)*np.exp(-self.kappa*t)
        calc_2kt = -0.25*self.sigma**2/self.kappa**3 *np.exp(-2.*self.kappa*t)
        if self.debug:
            print("@(t) %f Const %f T %f KT %f 2KT %f" % (t, const, calc_t, calc_kt, calc_2kt))
        return np.exp(-1.*(const + calc_t + calc_kt + calc_2kt))

#    def calc_norm_v(self, t0, t1):
#        ''' calculates vasicek norm (v(s, t0, t1) '''
#        res1 = self.sigma**2 / self.kappa**2
#        res2 = (np.exp(-self.kappa*t0) - np.exp(-self.kappa*t1))**2
#        res3 = 0.5*(np.exp(2*self.kappa*t0)-1.) / (self.kappa)
#        if self.debug:
#            print("self.kappa %f theta %f self.sigma %f t0 %f t1 %f" %
#                  (self.kappa, self.theta, self.sigma, t0, t1))
#            print("itm1 %f itm2 %f imt3 %f" % (res1, res2, res3))
#
#        return res1*res2*res3

    def calc_convexity_adjustment(self, t, t0, t1):
        ''' Calculates convexity adjustment '''
        res = -1.0
        delta = t1 - t0
        const = (self.sigma**2/self.kappa**3)*(np.exp(-self.kappa*t0) - np.exp(-self.kappa*t1))
        const2 = (np.exp(self.kappa*t0) - 0.5*np.exp(-self.kappa*t1)*np.exp(2.*self.kappa*t0))
        base = (np.exp(self.kappa*t) - 0.5*np.exp(-self.kappa*t1)*np.exp(2.*self.kappa*t))
        intermed = const*(const2 - base)
        pt0 = self.price_zero(t0)
        pt1 = self.price_zero(t1)
        ratio = pt0/(delta*pt1)
        res = ratio*(np.exp(intermed) + res)

        if self.debug:
            print("ratio %f Const %f Const2 %f Base %f Intermediate %f Result: %f" %
                  (ratio, const, const2, base, intermed, res))

        return res

    def calc_di_european_call(self, strike, t0, t1):
        ''' calculates d1 & d2 for european call on Zero coupon maturity t1 and exercise t0 '''
        norm_v = self.calc_norm_v(t=0, t0=t0, t1=t1)

        pt_t0 = self.price_zero(t0)
        pt_t1 = self.price_zero(t1)
        val = np.log(pt_t1/(strike*pt_t0))
        denom = np.sqrt(norm_v)
        d1 = (val + 0.5*norm_v)/denom
        d2 = (val - 0.5*norm_v)/denom
        if self.debug:
            print("calc_di_european_call: d1 %f d2 %f" % (d1, d2))

        return d1, d2

    def calc_european_put(self, strike, t0, t1):
        ''' calculates prices of euopean put on zero coupon bond maturity t1 and exerciose t0'''
        d1, d2 = self.calc_di_european_call(strike, t0, t1)

        p0 = self.price_zero(t0)
        p0 = p0*strike*norm_dist.cdf(-d2)

        p1 = self.price_zero(t1)
        p1 = p1*norm_dist.cdf(-d1)

        if self.debug:
            print("Price European Put p0 %f p1 %f price %f" % (p0, p1, (p0 - p1)))

        return p0 - p1

    def calc_di_caplet(self, strike, t1, t2, zero_t1, zero_t2, kappa=None, sigma=None):
        ''' calculates simple black d_i '''
        params = {}

        if sigma is not None and isinstance(sigma, float) and not np.isnan(sigma):
            params['sigma'] = sigma
        elif sigma is not None and isinstance(sigma, (list, np.ndarray)) and\
                all(np.logical_not(np.isnan(sigma))):
            params['sigma'] = sigma.copy()
        else:
            params['sigma'] = self.params['sigma']

        if kappa is not None and isinstance(kappa, float) and not np.isnan(kappa):
            params['kappa'] = kappa

        elif kappa is not None and isinstance(kappa, (list, np.ndarray)) and\
                all(np.logical_not(np.isnan(kappa))):
            params['kappa'] = kappa.copy()
        else:
            params['kappa'] = self.params['kappa']

        v_norm = self.calc_norm_v(t=0, t0=t1, t1=t2, params=params, dbg=self.debug)

        res = np.log((zero_t2/zero_t1)*(1. + (t2-t1)*strike))
        d1 = (res + 0.5*v_norm) / np.sqrt(v_norm)
        d2 = (res - 0.5*v_norm) / np.sqrt(v_norm)
        if self.debug:
            print("Di Caplet -- res %f v_norm %f d1 %f d2 %f " % (res, v_norm, d1, d2))

        return d1, d2

    def calc_price_caplet(self, strike, t1, t2, zero_t1, zero_t2, kappa=None, sigma=False):
        ''' calculates caplet price '''
        mult = 1. + (t2 - t1)*strike
        d1, d2 = self.calc_di_caplet(strike, t1, t2, zero_t1, zero_t2, kappa, sigma)
        p1 = zero_t1*norm_dist.cdf(-d2)

        p2 = mult*zero_t2*norm_dist.cdf(-d1)

        if self.debug:
            print("Caplet: p1 %f p2 %f cpl %f " % (p1, p2, (p1-p2)))

        return p1 - p2
