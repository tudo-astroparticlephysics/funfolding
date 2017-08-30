import numpy as np
from matplotlib import pyplot as plt

import funfolding as ff

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
    scale = 0.2

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

    model = ff.model.LinearModel()
    model.initialize(digitized_obs=binned_g,
                     digitized_truth=binned_f)
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

    vec_g, vec_f = model.generate_vectors(digitized_obs=binned_g,
                                          digitized_truth=binned_f)
    svd = ff.solution.SVDSolution()
    print('\n===========================\nResults for each Bin: Unfolded/True')

    print('\nSVD Solution for diffrent number of kept sigular values:')
    for i in range(1, 11):
        svd.initialize(model=model, vec_g=vec_g, tau=i)
        vec_f_est, V_f_est = svd.fit()
        str_0 = '{} singular values:'.format(str(i).zfill(2))
        str_1 = ''
        for f_i_est, f_i in zip(vec_f_est, vec_f):
            str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
        print('{}\t{}'.format(str_0, str_1))

    print('\nMCMC Solution: (constrained: sum(vec_f) == sum(vec_g)) : (FRIST RUN)')

    llh = ff.solution.StandardLLH(tau=None,
                                  C='thikonov',
                                  neg_llh=False)
    llh.initialize(vec_g=vec_g,
                   model=model)

    sol_mcmc = ff.solution.LLHSolutionMCMC(n_used_steps=2000,
                                           random_state=1337)
    sol_mcmc.initialize(llh=llh, model=model)
    sol_mcmc.set_x0_and_bounds()
    vec_f_est_mcmc, sigma_vec_f, samples, probs = sol_mcmc.fit()
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mcmc, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))

    print('\nMinimize Solution:')
    llh = ff.solution.StandardLLH(tau=None,
                                  C='thikonov',
                                  neg_llh=True)
    llh.initialize(vec_g=vec_g,
                   model=model)

    sol_mini = ff.solution.LLHSolutionMinimizer()
    sol_mini.initialize(llh=llh, model=model)
    sol_mini.set_x0_and_bounds()

    solution, V_f_est = sol_mini.fit(constrain_N=False)
    vec_f_est_mini = solution.x
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mini, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))

    print('\nMinimize Solution (constrained: sum(vec_f) == sum(vec_g)):')
    solution, V_f_est = sol_mini.fit(constrain_N=True)
    vec_f_est_mini = solution.x
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mini, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))

    print('\nMinimize Solution (MCMC as seed):')
    sol_mini.set_x0_and_bounds(x0=vec_f_est_mcmc)
    solution, V_f_est = sol_mini.fit(constrain_N=False)
    vec_f_est_mini = solution.x
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mini, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))

    corner.corner(samples, truths=vec_f)
    plt.savefig('corner_truth.png')
    print(np.sum(vec_f_est_mcmc))

    plt.clf()
    corner.corner(samples, truths=vec_f_est_mini, truth_color='r')
    plt.savefig('corner_mini.png')
    plt.clf()
    corner.corner(samples, truths=vec_f_est_mcmc, truth_color='springgreen')
    plt.savefig('corner_mcmc.png')
