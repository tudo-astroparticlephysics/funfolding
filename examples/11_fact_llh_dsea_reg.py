import logging
import os

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')


from funfolding import binning, model, solution



def generate_acceptance_correction(vec_f_truth,
                                   binning,
                                   logged_truth):
    e_min = 200
    e_max = 50000
    gamma = -2.7
    n_showers = 12000000
    if logged_truth:
        binning = np.power(10., binning)
    normalization = (gamma + 1) / (e_max ** (gamma + 1) - e_min ** (gamma + 1))
    corsika_cdf = lambda E: normalization * E ** (gamma + 1) / (gamma + 1)
    vec_acceptance = np.zeros_like(vec_f_truth, dtype=float)
    for i, vec_i_detected in enumerate(vec_f_truth):
        p_bin_i = corsika_cdf(binning[i + 1]) - corsika_cdf(binning[i])
        vec_acceptance[i] = p_bin_i * n_showers / vec_i_detected
    return vec_acceptance


if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

    random_seed = 1340
    n_walkers = 100
    n_steps_used = 2000
    n_samples_test = 5000
    min_samples_leaf = 20
    n_iterations = 20
    tau = 4.
    binning_E = np.linspace(2.4, 4.2, 10)

    random_state = np.random.RandomState(random_seed)
    if not os.path.isfile('fact_simulations.hdf'):
        from get_fact_simulations import download
        logging.info('Downloading FACT simulations!')
        download()
    df = pd.read_hdf('fact_simulations.hdf', 'gamma_simulation')

    binned_E = np.digitize(df.loc[:, 'log10(energy)'],
                           binning_E)

    idx = np.arange(len(df))
    random_state.shuffle(idx)

    test_slice = slice(0, n_samples_test)
    binning_slice = slice(n_samples_test, n_samples_test + 10 * n_samples_test)
    A_slice = slice(n_samples_test + 10 * n_samples_test, None)

    idx_test = np.sort(idx[test_slice])
    idx_binning = np.sort(idx[binning_slice])
    idx_A = np.sort(idx[A_slice])

    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(df.loc[:, 'log10(energy)'], binning_E)
    binned_E_test = binned_E[idx_test]
    binned_E_binning = binned_E[idx_binning]
    binned_E_A = binned_E[idx_A]

    tree_obs = ["log10(E_RF)",
                "log10(Size)",
                "log10(ConcCore)",
                "log10(numPixelInShower)",
                "log10(Length)",
                "Width",
                "M3Trans",
                "M3Long",
                "m3l",
                "m3t",
                "Concentration_onePixel",
                "Concentration_twoPixel",
                "Leakage",
                "Leakage2",
                "concCOG",
                "numIslands",
                "phChargeShower_mean",
                "phChargeShower_variance",
                "phChargeShower_max"]

    obs_array = df.get(tree_obs).values

    X_test = obs_array[idx_test]
    X_binning = obs_array[idx_binning]
    X_A = obs_array[idx_A]

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        min_samples_leaf=int(min_samples_leaf * 10.),
        random_state=random_state)

    tree_binning.fit(X_binning,
                     binned_E_binning,
                     uniform=False)

    binned_g_test = tree_binning.digitize(X_test)
    binned_g_binning = tree_binning.digitize(X_binning)
    binned_g_A = tree_binning.digitize(X_A)

    tree_model = model.LinearModel()
    tree_model.initialize(digitized_obs=binned_g_A,
                          digitized_truth=binned_E_A)
    vec_g_test, vec_f_test = tree_model.generate_vectors(
        digitized_obs=binned_g_test,
        digitized_truth=binned_E_test)

    vec_f_str = ', '.join('{0:.2f}'.format(a)
                          for a in vec_f_test)
    print('Truth:\t{}'.format(
        vec_f_str))

    dsea = solution.DSEAGaussianNB()
    dsea.initialize(X_A, binned_E_A)
    vec_f_dsea = dsea.fit(X_test)

    for i in range(n_iterations):
        vec_f_str = ', '.join('{0:.2f}'.format(a)
                              for a in vec_f_dsea)
        print('\tBest Fit (DSEA):\t{}'.format(
            vec_f_str))
        llh = solution.StandardLLH(tau=0.01,
                                   C='thikonov',
                                   reg_factor_f=vec_f_dsea)
        llh.initialize(vec_g=vec_g_test,
                       model=tree_model)

        sol_mini = solution.LLHSolutionMinimizer()
        sol_mini.initialize(llh=llh, model=tree_model)
        sol_mini.set_x0_and_bounds(x0=vec_f_dsea)
        best_fit = sol_mini.fit(constrain_N=False)[0]

        vec_f_str = ', '.join('{0:.2f}'.format(a)
                              for a in best_fit.x)
        print('\tBest Fit (Minimizer):\t{}\t(LLH: {})'.format(
            vec_f_str,
            best_fit.fun))

        priors = np.asarray(best_fit.x, dtype=np.float16)
        priors /= priors.sum()
        dsea = solution.DSEAGaussianNB()
        print(priors)
        try:
            dsea.initialize(X_A, binned_E_A, priors=priors)
        except ValueError:
            print(priors)
            print(priors.sum())
            exit()
        vec_f_dsea = dsea.fit(X_test)
