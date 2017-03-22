import logging
import numpy as np


class LinearModel:
    name = 'LinearModel'

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1
        self.A = None

    def initialize(self, X=None, y=None):
        self.logger.debug('Initilizing the model!')
        self.status = 0

    def evaluate(self, f=None):
        self.logger.debug('Model evaluation!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        return np.dot(self.A, f), f

    def set_x0(self, x0=None):
        self.logger.debug('Setup of x0!')
        return x0


class BasicLinearModel(LinearModel):
    name = 'BasicLinearModel'

    def __init__(self):
        super(BasicLinearModel, self).__init__()
        self.range_g = None
        self.range_f = None

    def __generate_binning__(self):
        binnings = []
        for r in [self.range_g, self.range_f]:
            low = r[0]
            high = r[-1]
            binnings.append(np.linspace(low, high + 1, high - low + 2))
        return binnings[0], binnings[1]

    def generate_vectors(self, g=None, f=None):
        binning_g, binning_f = self.__generate_binning__()
        if g is not None:
            vec_g = np.histogram(g, bins=binning_g)[0]
        else:
            vec_g = None
        if f is not None:
            vec_f = np.histogram(f, bins=binning_f)[0]
        else:
            vec_f = None
        return vec_g, vec_f

    def initialize(self, g, f, sample_weight=None):
        super(BasicLinearModel, self).initialize()
        self.range_g = (min(g), max(g))
        self.range_f = (min(f), max(f))
        binning_g, binning_f = self.__generate_binning__()

        self.A = np.histogram2d(x=g, y=f, bins=(binning_g, binning_f))[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=1))
        self.A = np.dot(M_norm, self.A)

    def generate_x0(self, vec_g):
        n = self.A.shape[1]
        return np.ones(n) * np.sum(vec_g) / n

    def generate_bounds(self, vec_g):
        n = self.A.shape[1]
        n_events = np.sum(vec_g)
        bounds = []
        for i in range(n):
            bounds.append((0, n_events))
        return bounds


class LinearModelConstantN(BasicLinearModel):
    name = 'LinearModelConstantN'

    def spherical_to_cart(self, f_spherical, N):
        a = np.concatenate((np.array([2 * np.pi]), f_spherical))
        si = np.sin(a)
        si[0] = 1
        si = np.cumprod(si)
        co = np.cos(a)
        co = np.roll(co, -1)
        f_cart = si * co * N
        f_cart[np.isclose(f_cart, 0.)] = 0.
        return f_cart

    def cart_to_spherical(self, f_cart):
        x = np.array(f_cart)
        f_spherical = np.zeros(len(f_cart) - 1)
        N = np.sqrt(np.dot(x.T, x))
        for i in range(len(f_spherical)):
            len_x = np.sqrt(np.dot(x[i:].T, x[i:]))
            if len_x == 0.:
                f_spherical[i] = 0.
            else:
                arg = x[i] / len_x
                f_spherical[i] = np.arccos(arg)
        if x[-1] < 0.:
            f_spherical[-1] = 2 * np.pi - f_spherical[-1]
        return f_spherical, N

    def generate_vectors(self, g=None, f=None):
        vec_g, vec_f = super(LinearModelConstantN, self).generate_vectors(
            g=g,
            f=f)
        if vec_f is None:
            N = None
            vec_f_spherical = None
        else:
            vec_f_spherical, N = self.cart_to_spherical(vec_f)
        return vec_g, vec_f_spherical, N

    def evaluate(self, f_spherical):
        f_cart = self.spherical_to_cart(f_spherical, self.N)
        return super(LinearModelConstantN, self).evaluate(f_cart)

    def set_x0(self, x0):
        super(LinearModelConstantN, self).set_x0()
        x0, N = self.cart_to_spherical(x0)
        self.N = N
        return x0

    def generate_bounds(self, vec_g):
        n = self.A.shape[1]
        bounds = []
        for i in range(n - 1):
            bounds.append((0, np.pi / 2.))
        return bounds
