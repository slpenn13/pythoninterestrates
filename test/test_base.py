''' testing routines correcpsonding to interest_rate_base.py '''
import pytest
import interest_rate_base as intbase


@pytest.mark.parametrize("tolerance,dbg", [(5.0e-5, True)])

def test_forward_d1_calc(tolerance, dbg):
    ''' forward test calculation '''
    t1 = 0.5
    t2 = 1.0
    zero_t1 = 0.970874
    zero_t2 = 0.933532
    x1 = intbase.calc_forward_rate_1d(t1, t2, zero_t1, zero_t2)
    if bool(dbg):
        print(x1, dbg)

    assert abs(x1 - 0.08) < tolerance
