import logging

import numpy as np
from matplotlib import pyplot as plt

import funfolding as ff

from scipy import integrate
from scipy.stats import norm
from scipy import linalg
from scipy.special import lambertw


def create_xexpax_sample(n_events, a, x_min, x_max):
    F = lambda x, N: -N * np.exp(-a * x) * (a * x + 1) / a**2
    N_x = 1 / (F(x_max, 1) - F(x_min, 1))
    f = lambda x: x * np.exp(-a * x) * N_x
    def F_inv(u):
        return -(1 + np.real(lambertw((-N_x + a**2 * u) / (np.e * N_x), k=-1))) / a
    u = np.random.uniform(size=n_events)
    return F_inv(u), f


def smear(x, factor=10, exponent=2, scale=5):
    x_loc = unbinned_x + (unbinned_x / x_max * factor)**exponent
    return np.random.normal(loc=x_loc, scale=scale)

if __name__ == '__main__':
    a = 0.2
    epsilon = 0.22
    x_min = 0.
    x_max = 20.
    n_bins = 20
    n_events_matrix = int(1e6)
    n_events_test = int(1e4)

    unbinned_x, f = create_xexpax_sample(
        n_events_matrix,
        a,
        x_min,
        x_max)

    unbinned_y = smear(unbinned_x)
    plt.hexbin(unbinned_x, unbinned_y, vmin=1)
    plt.xlabel('Sought-after Value')
    plt.ylabel('Measured Value')
    plt.savefig('02_x_y_smearing.png')
    binning_x = np.linspace(0, 20, 11)
    binning_y = np.linspace(min(unbinned_y) - 1e-3, max(unbinned_y) + 1e-3, 31)
    binned_x = np.digitize(unbinned_x, binning_x)
    binned_y = np.digitize(unbinned_y, binning_y)

    model = ff.model.BasicLinearModel()
    model.initialize(X=binned_x,
                     y=binned_y)
    plt.clf()
    plt.imshow(model.A)
    plt.savefig('02_matrix_A.png')

    unbinned_x, f = create_xexpax_sample(
        n_events_test,
        a,
        x_min,
        x_max)
    unbinned_y = smear(unbinned_x)
    binned_x = np.digitize(unbinned_x, binning_x)
    binned_y = np.digitize(unbinned_y, binning_y)

    vec_g, vec_f = model.generate_vectors(binned_x, binned_y)
    svd = ff.solution.SVDSolution()
    print(svd.run(vec_g, model))
