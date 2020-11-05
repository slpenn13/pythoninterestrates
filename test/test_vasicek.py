''' Pytest corresponding to calc_norm_v '''
# import numpy as np
import shortrate_model_vasicek as vas
import interest_rate_capfloor_convenience as intconv

mdl = vas.short_rate_vasicek(kappa=0.86, theta=0.08, sigma=0.01, r0=0.06,
                             norm_method=intconv.calc_v_norm_1d, dbg=True)

def test_norm():
    ''' init_ test'''
    assert round(1e6*mdl.calc_norm_v(t=0.0, t0=0.25, t1=0.5, dbg=True), 3) == 1.028

def test_pt_05():
    ''' initial pricing test '''
    assert round(mdl.price_zero(0.5), 4) == 0.9686

def test_put():
    ''' init test for put calculation '''
    x1 = mdl.price_zero(0.25)
    x2 = mdl.price_zero(0.5)
    assert round(1e4*mdl.calc_european_put(strike=(x2/x1), t0=0.25, t1=0.5), 3) == 3.918
