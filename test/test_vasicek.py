''' Pytest corresponding to calc_norm_v '''
import numpy as np
import shortrate_model_vasicek as vas

def calc_v_norm(t=0, t0=0, t1=0, params=None, dbg=False):
    ''' calculates v norm '''
    if params is None or 'sigma' not in params.keys():
        raise ValueError("No value for sigma provided")
    sigma = params['sigma']
    if params is None or 'kappa' not in params.keys():
        raise ValueError("No value for kappa provided")
    kappa = params['kappa']

    if params is None or 'theta' not in params.keys():
        raise ValueError("No value for theta provided")
    theta = params['theta']

    res1 = sigma**2/kappa**2*(np.exp(-kappa*t0) - np.exp(-kappa*t1))**2
    res2 = (np.exp(2.*kappa*t0) - 1.)/(2*kappa)


    if dbg:
        print("self.kappa %f theta %f self.sigma %f t %f t0 %f t1 %f" %
              (kappa, theta, sigma, t, t0, t1))
        print("itm1 %f itm2 %f" % (res1, res2))

    return res1*res2

mdl = vas.short_rate_vasicek(kappa=0.86, theta=0.08, sigma=0.01, r0=0.06,
                             norm_method=calc_v_norm, dbg=True)
def test_norm():
    ''' init_ test'''
    assert round(1e6*mdl.calc_norm_v(t=0.0, t0=0.25, t1=0.5, dbg=True), 3) == 1.028

def test_pt_05():
    ''' initila pricing test '''
    assert round(mdl.price_zero(0.5), 4) == 0.9686

def test_put():
    ''' init test for put calculation '''
    x1 = mdl.price_zero(0.25)
    x2 = mdl.price_zero(0.5)
    assert round(1e4*mdl.calc_european_put(strike=(x2/x1), t0=0.25, t1=0.5), 3) == 3.918
