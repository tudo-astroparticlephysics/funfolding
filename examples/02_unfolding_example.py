import logging

import numpy as np
from matplotlib import pyplot as plt

from IPython import embed

import funfolding as ff

from matplotlib import pyplot as plt

from scipy.special import lambertw
import corner

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

    exponent = 1
    scale = 2

    n_events_matrix = int(1e6)
    n_events_test = int(1e4)

    unbinned_f, f = create_xexpax_sample(
        n_events_matrix,
        a,
        x_min,
        x_max)

    unbinned_g = smear(unbinned_f, exponent=exponent, scale=scale)
    plt.hexbin(unbinned_g, unbinned_f, vmin=1)
    plt.xlabel('Sought-after Value')
    plt.ylabel('Measured Value')
    plt.savefig('02_x_y_smearing.png')
    print('\nScatter Plot of Sought-after vs. Measured Values saved as: '
          '02_x_y_smearing.png')

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
    print('\nNormalized Matrix saved as: 02_matrix_A.png')

    unbinned_f, f = create_xexpax_sample(
        n_events_test,
        a,
        x_min,
        x_max)
    unbinned_g = smear(unbinned_f, exponent=exponent, scale=scale)
    binned_g = np.digitize(unbinned_g, binning_g)
    binned_f = np.digitize(unbinned_f, binning_f)

    vec_g, vec_f = model.generate_vectors(binned_g, binned_f)
    print(vec_f)
    svd = ff.solution.SVDSolution()
    print('\n===========================\nResults for each Bin: Unfolded/True')

    print('\nSVD Solution for diffrent number of kept sigular values:')
    for i in range(1, 11):
        vec_f_est, V_f_est = svd.run(vec_g=vec_g,
                                     model=model,
                                     keep_n_sig_values=i)
        str_0 = '{} singular values:'.format(str(i).zfill(2))
        str_1 = ''
        for f_i_est, f_i in zip(vec_f_est, vec_f):
            str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
        print('{}\t{}'.format(str_0, str_1))

    print('\nMinimize Solution: (constrained: sum(vec_f) == sum(vec_g)) :')
    llh_sol = ff.solution.LLHSolutionMinimizer()
    llh_sol.initialize(vec_g=vec_g, model=model, bounds=True)
    vec_f_est_mini, V_f_est = llh_sol.run(tau=0)
    print(vec_f_est_mini)
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mini, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))

    print('\nMCMC Solution: (constrained: sum(vec_f) == sum(vec_g)) :')
    llh_mcmc = ff.solution.LLHSolutionMCMC(n_used_steps=5000,
                                           random_state=1337)
    llh_mcmc.initialize(vec_g=vec_g, model=model)
    vec_f_est_mcmc, sample = llh_mcmc.run(tau=0)
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mcmc, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))
    corner.corner(sample, truths=vec_f)
    plt.savefig('corner_truth.png')
    print(np.sum(vec_f_est_mcmc))
    exit()
    plt.clf()
    corner.corner(sample, truths=vec_f_est_mini, truth_color='r')
    plt.savefig('corner_mini.png')
    plt.clf()
    corner.corner(sample, truths=vec_f_est_mcmc, truth_color='springgreen')
    plt.savefig('corner_mcmc.png')
