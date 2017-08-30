import numpy as np

import funfolding as ff

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

    exponent = 1
    scale = 2

    n_pulls = 100

    n_events_matrix = int(1e6)
    n_events_test = int(1e4)

    unbinned_f, f = create_xexpax_sample(
        n_events_matrix,
        a,
        x_min,
        x_max)

    unbinned_g = smear(unbinned_f, exponent=exponent, scale=scale)

    binning_f = np.linspace(0, 20, 11)
    binning_g = np.linspace(min(unbinned_g) - 1e-3, max(unbinned_g) + 1e-3, 31)
    binned_g = np.digitize(unbinned_g, binning_g)
    binned_f = np.digitize(unbinned_f, binning_f)

    model = ff.model.LinearModel()
    model.initialize(digitized_obs=binned_g,
                     digitized_truth=binned_f)


    solutions = np.zeros((n_pulls, len(binning_f)-1))
    stds = np.zeros((n_pulls, len(binning_f)-1))
    quantils = np.zeros((n_pulls, len(binning_f)-1, 2))

    for i in range(n_pulls):
        print(i)
        unbinned_f, f = create_xexpax_sample(
            n_events_test,
            a,
            x_min,
            x_max)
        unbinned_g = smear(unbinned_f, exponent=exponent, scale=scale)
        binned_g = np.digitize(unbinned_g, binning_g)
        binned_f = np.digitize(unbinned_f, binning_f)
        vec_g, vec_f = model.generate_vectors(binned_g, binned_f)
        llh = ff.solution.StandardLLH(tau=None,
                                      C='thikonov',
                                      neg_llh=False)
        llh.initialize(vec_g=vec_g,
                       model=model)

        sol_mcmc = ff.solution.LLHSolutionMCMC()
        sol_mcmc.initialize(llh=llh, model=model)
        sol_mcmc.set_x0_and_bounds()
        vec_f_est_mcmc, sigma_vec_f, samples, probs = sol_mcmc.fit()
        print(vec_f_est_mcmc)
