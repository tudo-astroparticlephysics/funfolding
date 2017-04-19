import logging
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt


def ct(f_spherical, N):
    si = np.ones(len(f_spherical) + 1, dtype=float)
    co = np.ones(len(f_spherical) + 1, dtype=float)
    si[1:] = np.sin(f_spherical)
    co[:-1] = np.cos(f_spherical)
    si = np.cumprod(si)
    f_cart = si * co
    f_cart *= N / np.sum(f_cart)
    return f_cart


def ct_r(f):
    f_squared = f**2
    csum = np.cumsum(f_squared[::-1])[::-1]
    f_spherical = np.zeros(len(f) - 1)
    f_spherical = np.arccos(f[:-1] / np.sqrt(csum[:-1]))
    N = np.sum(f)
    return f_spherical, N


class LinearModel:
    name = 'LinearModel'

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1
        self.A = None
        self.dim_f = None

    def initialize(self, X=None, y=None):
        self.logger.debug('Initilizing the model!')
        self.status = 0

    def evaluate(self, f=None):
        self.logger.debug('Model evaluation!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        return np.dot(self.A, f), f, f

    def set_x0(self, x0=None):
        self.logger.debug('Setup of x0!')
        return x0

    def transform(self, f):
        self.logger.debug('Transforming f!')
        return f


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
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = np.dot(self.A, M_norm)
        self.dim_f = self.A.shape[1]

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

    def evaluate_condition(self, ax=None, label='Linear Model'):
        self.logger.debug('Evaluation of Singular Values!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        U, S_values, V = linalg.svd(self.A)
        if ax is None:
            _, ax = plt.subplots()
        ax.set_xlabel(r'Index $j$')
        ax.set_ylabel(r'Normed Singular Values $\frac{\lambda_i}{\lambda_0}$')

        S_values = S_values / S_values[0]
        binning = np.linspace(-0.5,
                              len(S_values) - 0.5,
                              len(S_values) + 1)
        x_pos = np.arange(len(S_values))
        ax.hist(x_pos,
                bins=binning,
                weights=S_values,
                histtype='step',
                label=label)
        ax.set_xlim([binning[0], binning[-1]])
        return ax


class MCBiasLinearModel(BasicLinearModel):
    name = 'MCBiasLinearModel'

    def __init__(self):
        super(MCBiasLinearModel, self).__init__()
        self._vec_f_MC = None
        self.correction = 1.

    @property
    def vec_f_MC(self):
        return self._vec_f_MC

    @vec_f_MC.setter
    def vec_f_MC(self, value):
        self._vec_f_MC = value

    def evaluate(self, f_0):
        f = self.vec_f_MC * f_0 * self.correction
        g, f, _ = super(MCBiasLinearModel, self).evaluate(f)
        return g, f, f_0

    def generate_x0(self, vec_g):
        self.correction = np.sum(vec_g) / np.sum(self._vec_f_MC)
        return np.ones_like(self._vec_f_MC)

    def transform(self, f_0):
        return f_0 * self._vec_f_MC


class SphericalLinearModel(BasicLinearModel):
    name = 'SphericalLinearModel'

    def spherical_to_cart(self, f_spherical):
        return ct(f_spherical, self.N)

    def cart_to_spherical(self, f_cart):
        return ct_r(f_cart)

    def initialize(self, g, f, sample_weight=None):
        super(SphericalLinearModel, self).initialize(
            g=g,
            f=f,
            sample_weight=sample_weight)
        self.dim_f -= 1

    def generate_vectors(self, g=None, f=None):
        vec_g, vec_f = super(SphericalLinearModel, self).generate_vectors(
            g=g,
            f=f)
        if vec_f is None:
            N = None
            vec_f_spherical = None
        else:
            vec_f_spherical, N = self.cart_to_spherical(vec_f)
        return vec_g, vec_f_spherical, N

    def evaluate(self, f_spherical):
        print(f_spherical)
        f_cart = self.spherical_to_cart(f_spherical)
        print(f_cart)
        return super(SphericalLinearModel, self).evaluate(f_cart)

    def set_x0(self, x0):
        super(SphericalLinearModel, self).set_x0()
        x0, N = self.cart_to_spherical(x0)
        self.N = N
        return x0

    def generate_bounds(self, vec_g):
        n = self.A.shape[1]
        bounds = []
        for i in range(n - 1):
            bounds.append((0, np.pi / 2.))
        return bounds
