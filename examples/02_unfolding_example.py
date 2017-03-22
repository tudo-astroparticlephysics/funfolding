import logging

import numpy as np
from matplotlib import pyplot as plt

import funfolding as ff

from scipy import integrate
from scipy.stats import norm
from scipy import linalg
from scipy.special import lambertw

seed = 1337

rnd = np.random.RandomState(seed)


def create_xexpax_sample(n_events, a, x_min, x_max):
    def F(x, N):
        return -N * np.exp(-a * x) * (a * x + 1) / a**2
    N_x = 1 / (F(x_max, 1) - F(x_min, 1))

    def f(x):
        return x * np.exp(-a * x) * N_x

    def F_inv(u):
        prod_log = np.real(lambertw((-N_x + a**2 * u) / (np.e * N_x), k=-1))
        return -(1 + prod_log) / a
    u = rnd.uniform(size=n_events)
    return F_inv(u), f


def smear(unbinned_f, factor=10, exponent=2, scale=5):
    x_loc = unbinned_f + (unbinned_f / x_max * factor)**exponent
    return rnd.normal(loc=x_loc, scale=scale)


if __name__ == '__main__':

    a = 0.2
    x_min = 0.
    x_max = 20.

    n_events_matrix = int(1e6)
    n_events_test = int(1e4)

    unbinned_f, f = create_xexpax_sample(
        n_events_matrix,
        a,
        x_min,
        x_max)

    unbinned_g = smear(unbinned_f, exponent=1, scale=0.5)
    plt.hexbin(unbinned_g, unbinned_f, vmin=1)
    plt.xlabel('Sought-after Value')
    plt.ylabel('Measured Value')
    plt.savefig('02_x_y_smearing.png')
    binning_f = np.linspace(0, 20, 11)
    binning_g = np.linspace(min(unbinned_g) - 1e-3, max(unbinned_g) + 1e-3, 31)
    binned_g = np.digitize(unbinned_g, binning_g)
    binned_f = np.digitize(unbinned_f, binning_f)

    model = ff.model.BasicLinearModel()
    model.initialize(g=binned_g,
                     f=binned_f)
    plt.clf()
    plt.imshow(model.A)
    plt.savefig('02_matrix_A.png')

    model_const_N = ff.model.LinearModelConstantN()
    model_const_N.initialize(g=binned_g,
                             f=binned_f)

    unbinned_f, f = create_xexpax_sample(
        n_events_test,
        a,
        x_min,
        x_max)
    unbinned_g = smear(unbinned_f, exponent=1, scale=0.5)
    binned_g = np.digitize(unbinned_g, binning_g)
    binned_f = np.digitize(unbinned_f, binning_f)

    vec_g, vec_f = model.generate_vectors(binned_g, binned_f)
    svd = ff.solution.SVDSolution()
    for i in range(1, 12):
        print('svd {} sig_vals:'.format(i))
        print(list(svd.run(vec_g, model, i)[0] / vec_f))

    vec_g, vec_f_0, N = model_const_N.generate_vectors(binned_g, binned_f)

    llh_sol = ff.solution.LLHSolutionMinimizer()
    solution = llh_sol.run(vec_g=vec_g,
                           model=model,
                           tau=0,
                           bounds=True)

    print(solution)
    solution = llh_sol.run(vec_g=vec_g,
                           model=model_const_N,
                           tau=0,
                           bounds=True)
    print(solution)
