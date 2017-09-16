import logging

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from funfolding import binning, model, solution, pipeline


def do_single_pull(obs_array_binning,
                   obs_array_A,
                   obs_array_test,
                   y_binning,
                   y_A,
                   y_test,
                   random_state=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    logging.info('\t{} Events for testing'.format(len(y_test)))
    logging.info('\t{} Events for binning'.format(len(y_binning)))
    logging.info('\t{} Events for A'.format(len(y_A)))

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        min_samples_leaf=min_samples_leaf * 10,
        random_state=random_state)
    tree_binning.fit(obs_array_binning,
                     y_binning)
    binned_g_A = tree_binning.digitize(obs_array_A)
    binned_g_test = tree_binning.digitize(obs_array_test)
    tree_model = model.LinearModel()
    tree_model.initialize(digitized_obs=binned_g_A,
                          digitized_truth=y_A)

    vec_g, _ = tree_model.generate_vectors(digitized_obs=binned_g_test)

    llh = solution.StandardLLH(tau=None,
                               C='thikonov')
    llh.initialize(vec_g=vec_g,
                   model=tree_model)

    llh_truth = llh.evaluate_llh(vec_f_truth)

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

    vec_f_str = ', '.join('{0:.2f}'.format(a)
                          for a in vec_f_est_mcmc)
    logging.info('\tBest Fit (MCMC):\t{}\t(LLH: {})'.format(
                 vec_f_str,
                 max(probs)))
    return float(np.sum(probs < llh_truth)) / len(probs)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-j", "--n_jobs",
                      dest='n_jobs',
                      default=1)
    (options, args) = parser.parse_args()

    random_seed = 1338
    n_pulls = 5000
    n_walker = 100
    n_steps_used = 2000
    n_samples_test = 5000
    min_samples_leaf = 20
    binning_E = np.linspace(2.4, 4.2, 10)

    n_jobs = int(options.n_jobs)

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

    vec_f_truth = np.array(np.bincount(binned_E), dtype=float)[1:]
    vec_f_truth /= np.sum(vec_f_truth)
    vec_f_truth *= n_samples_test

    vec_f_truth_str = ', '.join('{0:.2f}'.format(a)
                                for a in vec_f_truth)
    if n_jobs == 1:
        logging.captureWarnings(True)
        logging.basicConfig(
            format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
            level=logging.INFO)
        logging.info('Underlying Spectrum: {}'.format(vec_f_truth_str))
    else:
        print('Underlying Spectrum: {}'.format(vec_f_truth_str))
        print('Doing {} Pulls in max. {} parallel Jobs!'.format(n_pulls,
                                                                n_jobs))
    if n_jobs > 1:
        import time
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            p_values = []

            def future_callback(future):
                future_callback.finished += 1
                print('{}/{} Pulls finished!'.format(
                    future_callback.finished, n_pulls))
                if not future.cancelled():
                    p_value = future.result()
                    p_values.append(p_value)
                else:
                    p_values.append(None)
                future_callback.running -= 1

            future_callback.running = 0
            future_callback.finished = 0

            for i, (idx_test, idx_A, idx_binning) in enumerate(
                    pull_mode_iterator):
                while True:
                    if future_callback.running < n_jobs:
                        break
                    else:
                        time.sleep(1)
                future = executor.submit(
                    do_single_pull,
                    obs_array_binning=obs_array[idx_binning],
                    obs_array_A=obs_array[idx_A],
                    obs_array_test=obs_array[idx_test],
                    y_binning=binned_E[idx_binning],
                    y_A=binned_E[idx_A],
                    y_test=binned_E[idx_test],
                    random_state=random_seed + i)
                future.add_done_callback(future_callback)
                future_callback.running += 1
    else:
        p_values = np.zeros(n_pulls)
        for i, (idx_test, idx_A, idx_binning) in enumerate(pull_mode_iterator):
            logging.info('{}/{} started'.format(i + 1, n_pulls))
            p_values[i] = do_single_pull(
                obs_array_binning=obs_array[idx_binning],
                obs_array_A=obs_array[idx_A],
                obs_array_test=obs_array[idx_test],
                y_binning=binned_E[idx_binning],
                y_A=binned_E[idx_A],
                y_test=binned_E[idx_test],
                random_state=random_seed + i)
            logging.info('\tP-Value: {0:.2f}'.format(p_values[i]))

    fig, ax = plt.subplots()
    ax.hist(p_values, bins=10)
    ax.set_xlabel('P Values')
    ax.set_ylabel('Frequency')
    fig.savefig('07_p_values.png')
