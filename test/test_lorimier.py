''' testing routines correcpsonding to interest_rate_base.py '''
import json
import pytest
import curve_constructor_lorimier as cvl
import ir_discount_review as rvw

@pytest.fixture
def tolerance():
    return 5.0e-3

class loader():
    ''' loader -- setup for testing '''
    def __init__(self, json_file='./data/lorimier.json', dbg=False):
        ''' json file specifying run condictions and debug'''
        with open(json_file, "r") as fp:
            self.options = json.load(fp)
        fp.close()
        self.dbg = dbg

        self.lorimier = cvl.curve_builder_lorimier(self.options, alpha=0.1, dbg=True)
        self.rvw_disc = rvw.discount_calculator_review(
            self.options["review"], self.lorimier.zeros, dbg=dbg)
        self.rvw_disc.review()
        if self.dbg:
            print(self.lorimier.zeros.matrix)


@pytest.fixture
def strt():
    return loader()

def test_yield_time_6(strt, tolerance):
    ''' test cap solver ealuvator 1 '''

    at6 = strt.lorimier.zeros.f0(6.0, dbg=False)
    assert abs(-at6 + -0.41) < tolerance

def test_min_error(strt, tolerance):
    ''' test cap solver ealuvator 2 '''
    assert strt.rvw_disc.results_stats['yield_diff']['min'] < tolerance


def test_25_perc(strt, tolerance):
    ''' test cap solver ealuvator 5 '''
    assert abs(strt.rvw_disc.results_stats['yield_diff']['25%']) > tolerance

def test_alpha_5_results(tolerance):
    json_file = './data/lorimier.json'
    with open(json_file, "r") as fp:
        options = json.load(fp)
    fp.close()

    lorimier = cvl.curve_builder_lorimier(options, alpha=5.0, dbg=False)
    rvw_disc = rvw.discount_calculator_review(
        options["review"], lorimier.zeros, dbg=False)
    rvw_disc.review()

    assert rvw_disc.results_stats['yield_diff']['max'] < tolerance

class loader2():
    ''' loader -- setup for testing '''
    def __init__(self, json_file='./data/lorimier.json', dbg=False):
        ''' json file specifying run condictions and debug'''
        with open(json_file, "r") as fp:
            self.options = json.load(fp)
        fp.close()
        self.dbg = dbg
        self.options['data']['file'] =\
            '/home/spennington/git/pythoninterestrates/test/data/lorimier_swiss_repeated.csv'

        self.lorimier = cvl.curve_builder_lorimier(self.options, alpha=0.1, dbg=False)
        self.rvw_disc = rvw.discount_calculator_review(
            self.options["review"], self.lorimier.zeros, dbg=dbg)
        self.rvw_disc.review()
        if self.dbg:
            print(self.lorimier.zeros.matrix)


@pytest.fixture
def strt2():
    return loader2()

def test_yield_time_62(strt2, tolerance):
    ''' test cap solver ealuvator 1 '''

    at6 = strt2.lorimier.zeros.f0(6.0, dbg=False)
    assert abs(-at6 + -0.41) < tolerance

def test_min_error_2(strt2, tolerance):
    ''' test cap solver ealuvator 2 '''
    assert strt2.rvw_disc.results_stats['yield_diff']['min'] < tolerance
