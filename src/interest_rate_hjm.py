''' Common pricing methods corresponding to Interest rate Instruments '''


class hjm_model():
    ''' base class corresponding to mean reverting short rate model '''

    def __init__(self, kappa, theta, sigma, r0, calc_norm_method=None):
        ''' constructor
        '''
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0
        self.params = {'kappa': self.kappa, 'theta': self.theta, 'sigma': sigma, 'r0': self.r0}

        if calc_norm_method is None:
            raise ValueError("No calc norm method provided")
        self.calc_norm_method = calc_norm_method

    def __repr__(self):
        ''' print elements of short rate_model '''
        if isinstance(self.kappa, float) and isinstance(self.sigma, float):
            print("Kappa %f Theta %f Sigma %f r0 %f" %(self.kappa, self.theta, self.sigma,
                                                       self.r0))
        else:
            print("Theta %f r0 %f " % (self.theta, self.r0))
            print("Kappa")
            print(self.kappa)
            print("Sigma")
            print(self.sigma)

    def calc_norm_v(self, **kwargs):
        ''' calc norm method '''
        t = (kwargs['t'] if 't' in kwargs.keys() else 0.0)
        t0 = (kwargs['t0'] if 't0' in kwargs.keys() else 0.0)
        t1 = (kwargs['t1'] if 't1' in kwargs.keys() else 0.0)
        dbg = (kwargs['dbg'] if 'dbg' in kwargs.keys() else False)

        if 'params' in kwargs.keys():
            if dbg:
                print("params found ")
            return  self.calc_norm_method(t=t, t0=t0, t1=t1, params=kwargs['params'], dbg=dbg)

        return self.calc_norm_method(t=t, t0=t0, t1=t1, params=self.params, dbg=dbg)
