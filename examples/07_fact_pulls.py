import logging
import cPickle

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from funfolding import binning, model, solution, pipeline


if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

    random_seed = 1338

    n_pulls = 500
    n_walker = 100
    n_steps_used = 2000
    n_samples_test = 5000
    min_samples_leaf = 20
    binning_E = np.linspace(2.4, 4.2, 10)


    df = pd.read_hdf('fact_simulations.hdf', 'gamma_simulation')

    binned_E = np.digitize(df.loc[:, 'log10(energy)'],
                           binning_E)

    random_state = np.random.RandomState(random_seed)

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

    pull_mode_iterator = pipeline.split_test_unfolding(
        n_iterations=n_pulls,
        n_events_total=len(obs_array),
        n_events_test=n_samples_test,
        n_events_A=-1,
        n_events_binning=n_samples_test * 10.,
        random_state=random_state)

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=min_samples_leaf*10,
        random_state=random_state)

    llh_samples = np.zeros((n_pulls, n_steps_used * n_walker))
    llh_truth = np.zeros(n_pulls)
    best_fit = np.zeros((n_pulls, len(binning_E) - 1))
    p_values = np.zeros(n_pulls)

    vec_f_truth = np.array(np.bincount(binned_E), dtype=float)[1:]
    vec_f_truth /= np.sum(vec_f_truth)
    vec_f_truth *= n_samples_test

    vec_f_truth_str = ', '.join('{0:.2f}'.format(a)
                               for a in vec_f_truth)
    logging.info('Underlying Spectrum: {}'.format(vec_f_truth_str))

    for i, (idx_test, idx_A, idx_binning) in enumerate(pull_mode_iterator):
        logging.info('{}/{} started'.format(i + 1, n_pulls))
        logging.info('\t{} Events for testing'.format(len(idx_test)))
        logging.info('\t{} Events for binning'.format(len(idx_binning)))
        logging.info('\t{} Events for A'.format(len(idx_A)))
        tree_binning.fit(obs_array[idx_binning],
                         binned_E[idx_binning])
        binned_g_A = tree_binning.digitize(obs_array[idx_A])
        binned_g_test = tree_binning.digitize(obs_array[idx_test])
        tree_model = model.LinearModel()
        tree_model.initialize(digitized_obs=binned_g_A,
                              digitized_truth=binned_E[idx_A])

        vec_g, _ = tree_model.generate_vectors(digitized_obs=binned_g_test)

        llh = solution.StandardLLH(tau=None,
                                   C='thikonov')
        llh.initialize(vec_g=vec_g,
                       model=tree_model)

        llh_truth[i] = llh.evaluate_llh(vec_f_truth)

        sol_gd = solution.LLHSolutionGradientDescent(n_steps=500,
                                                     gamma=0.01)
        sol_gd.initialize(llh=llh, model=tree_model)
        sol_gd.set_x0_and_bounds()
        x, llh_values, gradient, hessian = sol_gd.fit()
        idx_best = np.argmax(llh_values)
        vec_f_str = ', '.join('{0:.2f}'.format(a)
                              for a in x[idx_best])
        logging.info('\tBest Fit (Gradient):\t{}\t(LLH: {})'.format(
            vec_f_str,
            llh_values[idx_best]))
        logging.info('\tStarting MCMC sampling...')
        sol_mcmc = solution.LLHSolutionMCMC(n_burn_steps=500,
                                            n_used_steps=2000,
                                            n_walker=100,
                                            random_state=random_state)
        sol_mcmc.initialize(llh=llh, model=tree_model)
        sol_mcmc.set_x0_and_bounds(x0=x[idx_best])
        vec_f_est_mcmc, sigma_vec_f, sample, probs = sol_mcmc.fit()
        best_fit[i, :] = vec_f_est_mcmc

        llh_samples[i, :] = probs

        vec_f_str = ', '.join('{0:.2f}'.format(a)
                              for a in vec_f_est_mcmc)
        logging.info('\tBest Fit (MCMC):\t{}\t(LLH: {})'.format(
            vec_f_str,
            max(probs)))

        p_values[i] = float(np.sum(probs < llh_truth[i])) / len(probs)
        logging.info('\tP-Value: {0:.2f}'.format(p_values[i]))

    fig, ax = plt.subplots()
    ax.hist(p_values)
    ax.set_xlabel('P Values')
    ax.set_ylabel('Frequency')
    fig.savefig('07_p_values.png')
