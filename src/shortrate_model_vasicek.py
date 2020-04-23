''' Implementation of vasicek class incorporating key calcultions '''
# import businessdate as bdte
import numpy as np
from scipy.stats import norm as norm_dist
# import pandas as pd
from interest_rate_utilities import short_rate_model as  short_rate


class short_rate_vasicek(short_rate):
    ''' implementation of Vasicek Short Rate Model '''
    def __init__(self, kappa, theta, sigma, r0, dbg=False):
        ''' Vasicek implementation of short rate model '''
        super().__init__(kappa, theta, sigma, r0)
        self.debug = dbg

        if self.debug:
            print("kappa %f theta %f sigma %f r0 %f" % (kappa, theta, sigma, r0))

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

    def calc_norm_v(self, t0, t1):
        ''' calculates vasicek norm (v(s, t0, t1) '''
        res1 = self.sigma**2 / self.kappa**2
        res2 = (np.exp(-self.kappa*t0) - np.exp(-self.kappa*t1))**2
        res3 = 0.5*(np.exp(2*self.kappa*t0)-1.) / (self.kappa)
        if self.debug:
            print("self.kappa %f theta %f self.sigma %f t0 %f t1 %f" %
                  (self.kappa, self.theta, self.sigma, t0, t1))
            print("itm1 %f itm2 %f imt3 %f" % (res1, res2, res3))

        return res1*res2*res3

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
        norm_v = self.calc_norm_v(t0, t1)

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
