import logging
import os

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.colors as colors


from funfolding import binning, model, solution
from funfolding.visualization import visualize_classic_binning
from funfolding.visualization import visualize_tree_binning
from funfolding.visualization import visualize_model
from funfolding.visualization.visualize_llh import plot_llh_slice

import corner
from scipy import linalg


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
    tau = 1.
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

    obs_array = df.get(['log10(ConcCore)', 'log10(E_RF)']).values

    X_A = obs_array[idx_A]
    X_binning = obs_array[idx_binning]
    X_test = obs_array[idx_test]

    classic_binning = binning.ClassicBinning(
        bins=[15, 25],
        random_state=random_state)

    classic_binning.fit(X_A)

    fig, ax = plt.subplots()
    visualize_classic_binning.plot_binning(ax,
                 classic_binning,
                 X_A,
                 log_c=False,
                 cmap='viridis')
    fig.savefig('05_fact_example_original_binning.png')




    tree_binning_2d = binning.TreeBinningSklearn(
        regression=False,
        min_samples_leaf=int(min_samples_leaf * 10.),
        random_state=random_state)

    tree_binning_2d.fit(X_binning,
                     binned_E_binning,
                     uniform=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    visualize_tree_binning.plot_binning(ax,
                                        tree_binning_2d,
                                        limits=[-0.7,
                                                -0.2,
                                                2.7,
                                                4.2],
                                        X=X_A,
                                        linecolor='k',
                                        linewidth='0.3',
                                        log_c=False,
                                        as_hexbins=True,
                                        hex_kwargs={'rasterized': True,
                                                    'gridsize': 50},
                                        cmap='viridis')
    ax.set_ylabel('log10(Energy Estimator [GeV])')
    ax.set_xlabel('log10(Concentration [a.u.])')
    fig.savefig('05_fact_example_original_tree_binning.png', dpi=300)




    closest = classic_binning.merge(X_binning,
                                    min_samples=int(min_samples_leaf * 10.),
                                    mode='closest')
    fig, ax = plt.subplots()
    visualize_classic_binning.plot_binning(ax,
                 closest,
                 X_A,
                 log_c=False,
                 cmap='viridis')
    fig.savefig('05_fact_example_original_binning_closest.png')


    unmerged_model = model.LinearModel()
    binned_g_A = classic_binning.digitize(X_A)
    unmerged_model.initialize(digitized_obs=binned_g_A,
                              digitized_truth=binned_E_A)

    binned_g_A = closest.digitize(X_A)
    merged_model = model.LinearModel()
    merged_model.initialize(digitized_obs=binned_g_A,
                            digitized_truth=binned_E_A)

    single_obs_model = model.LinearModel()
    max_e = np.max(X_A[:, 1]) + 1e-3
    min_e = np.min(X_A[:, 1]) - 1e-3
    binning_E_obs = np.linspace(min_e, max_e, 11)
    binned_g_A = np.digitize(X_A[:, 1], binning_E_obs)
    single_obs_model.initialize(digitized_obs=binned_g_A,
                                digitized_truth=binned_E_A)

    single_obs_model_more_bins = model.LinearModel()
    max_e = np.max(X_A[:, 1]) + 1e-3
    min_e = np.min(X_A[:, 1]) - 1e-3
    binning_E_obs = np.linspace(min_e, max_e, closest.n_bins + 1)
    binned_g_A = np.digitize(X_A[:, 1], binning_E_obs)
    single_obs_model_more_bins.initialize(digitized_obs=binned_g_A,
                                          digitized_truth=binned_E_A)
    fig, ax = plt.subplots(figsize=(2, 6))
    visualize_model.plot_A(ax, merged_model)
    fig.savefig('05_A_single_obs_model.png', dpi=300)

    binned_g_A = tree_binning_2d.digitize(X_A)
    tree_2d_model = model.LinearModel()
    tree_2d_model.initialize(digitized_obs=binned_g_A,
                            digitized_truth=binned_E_A)
    fig, ax = plt.subplots(figsize=(2, 6))
    visualize_model.plot_A(ax, tree_2d_model)
    fig.savefig('05_A_tree_model.png', dpi=300)

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

    X_tree_test = obs_array[idx_test]
    X_tree_binning = obs_array[idx_binning]
    X_tree_A = obs_array[idx_A]

    tree_binning_uniform = binning.TreeBinningSklearn(
        regression=False,
        min_samples_leaf=int(min_samples_leaf * 10.),
        random_state=random_state)

    tree_binning_uniform.fit(X_tree_binning,
                             binned_E_binning,
                             uniform=True)

    binned_g_A = tree_binning_uniform.digitize(X_tree_A)

    tree_model_uniform = model.LinearModel()
    tree_model_uniform.initialize(digitized_obs=binned_g_A,
                                  digitized_truth=binned_E_A)
    fig, ax = plt.subplots(figsize=(2, 6))
    visualize_model.plot_A(ax, tree_model_uniform)
    fig.savefig('05_A_tree_model_full_uniform.png', dpi=300)

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        min_samples_leaf=int(min_samples_leaf * 10.),
        random_state=random_state)

    tree_binning.fit(X_tree_binning,
                     binned_E_binning,
                     uniform=False)

    binned_g_A = tree_binning.digitize(X_tree_A)

    tree_model = model.LinearModel()
    tree_model.initialize(digitized_obs=binned_g_A,
                          digitized_truth=binned_E_A)
    visualize_model.plot_A(ax, tree_model)
    fig.savefig('05_A_tree_model_full.png', dpi=300)


    fig, ax = plt.subplots()
    svd_values = unmerged_model.evaluate_condition()
    bin_edges = np.linspace(0, len(svd_values), len(svd_values) + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='2 Observables (Unmerged; {} Bins)'.format(
                classic_binning.n_bins))

    svd_values = merged_model.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='2 Observables (Merged; {} Bins)'.format(closest.n_bins))
#
#    svd_values = single_obs_model.evaluate_condition()
#    ax.hist(bin_centers,
#            bins=bin_edges,
#            weights=svd_values,
#            histtype='step',
#            label='Single Observable ({} Bins)'.format(closest.n_bins))
#
#    svd_values = single_obs_model_more_bins.evaluate_condition()
#    ax.hist(bin_centers,
#            bins=bin_edges,
#            weights=svd_values,
#            histtype='step',
#            label='Single Observable ({} Bins)'.format(closest.n_bins))

    svd_values = tree_2d_model.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='Tree Based 2D ({} Bins)'.format(tree_binning.n_bins))


    svd_values = tree_model.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='Tree Based ({} Bins)'.format(tree_binning.n_bins))
#
    svd_values = tree_model_uniform.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
             weights=svd_values,
            histtype='step',
            label='Tree Based ({} Bins; Uniform)'.format(tree_binning.n_bins))

    plt.legend(loc='lower left')
    ax.set_yscale("log", nonposy='clip')
    plt.savefig('05_condition.png')

    exit()

    binned_g_test = tree_binning.digitize(X_tree_test)
    vec_g, vec_f = tree_model.generate_vectors(binned_g_test,
                                               binned_E_test)

    vec_f_truth = np.array(np.bincount(binned_E), dtype=float)[1:]
    vec_acceptance = generate_acceptance_correction(vec_f_truth,
                                                    binning_E,
                                                    True)
    vec_f_truth /= np.sum(vec_f_truth)
    vec_f_truth *= n_samples_test

    llh = solution.StandardLLH(tau=tau,
                               log_f=True,
                               vec_acceptance=vec_acceptance,
                               C='thikonov')
    llh.initialize(vec_g=vec_g,
                   model=tree_model)

    sol_gd = solution.LLHSolutionGradientDescent(n_steps=500,
                                                 gamma=0.01)
    sol_gd.initialize(llh=llh, model=tree_model)
    sol_gd.set_x0_and_bounds()
    x, llh_values, gradient, hessian = sol_gd.fit()
    idx_best = np.argmax(llh_values)
    vec_f_str = ', '.join('{0:.2f}'.format(a)
                          for a in x[idx_best])
    logging.info('Best Fit (Gradient):\t{}\t(LLH: {})'.format(
        vec_f_str,
        llh_values[idx_best]))

    sol_mini = solution.LLHSolutionMinimizer()
    sol_mini.initialize(llh=llh, model=tree_model)
    sol_mini.set_x0_and_bounds(x0=x[idx_best])
    best_fit, mini_cov = sol_mini.fit(constrain_N=False)

    vec_f_str = ', '.join('{0:.2f}'.format(a)
                          for a in best_fit.x)
    logging.info('Best Fit (Minimizer):\t{}\t(LLH: {})'.format(
        vec_f_str,
        best_fit.fun))

    sol_mcmc = solution.LLHSolutionMCMC(n_burn_steps=100,
                                        n_used_steps=n_steps_used,
                                        n_walkers=n_walkers,
                                        n_threads=1,
                                        random_state=random_state)
    sol_mcmc.initialize(llh=llh, model=tree_model)
    sol_mcmc.set_x0_and_bounds(x0=best_fit.x)
    vec_f_est_mcmc, sigma_vec_f, sample, probs = sol_mcmc.fit()

    vec_f_str = ', '.join('{0:.2f}'.format(a)
                          for a in vec_f_est_mcmc)
    logging.info('Best Fit (MCMC):\t{}\t(LLH: {})'.format(
        vec_f_str,
        max(probs)))

    # sol_mcmc.n_threads = 9
    # logging.info('Calculating Eff sample size:')
    # n_eff = sol_mcmc.calc_effective_sample_size(sample, n_threads=9)
    # n_eff_str = ', '.join(str(n) for n in n_eff)
    # logging.info('per Walker:\t{} ({} Walker with {} steps)'.format(
    #     n_eff_str,
    #     n_walkers,
    #     n_steps_used))

    def create_llh_slice(llh, best_fit, selected_bin=None):
        if selected_bin is None:
            selected_bin = np.argmax(best_fit)
        points = np.linspace(0.9 * best_fit[selected_bin],
                             1.1 * best_fit[selected_bin],
                             31)
        llh_values = np.zeros_like(points)
        gradient_values = np.zeros_like(points)
        hessian_values = np.zeros_like(points)

        fig, [ax_grad, ax_hess] = plt.subplots(2, 1, figsize=(24, 18))
        diff = np.diff(points)[0] / 1.5
        for i, p_i in enumerate(points):
            best_fit[selected_bin] = p_i
            llh_values[i] = llh.evaluate_llh(best_fit)
            gradient_values[i] = llh.evaluate_gradient(best_fit)[selected_bin]
            hessian_values[i] = llh.evaluate_hessian(best_fit)[selected_bin,
                                                               selected_bin]
            lower_x = p_i - diff
            upper_x = p_i + diff

            grad_lower_y = llh_values[i] - (diff * gradient_values[i])
            grad_upper_y = llh_values[i] + (diff * gradient_values[i])

            hess_lower_y = gradient_values[i] - (diff * hessian_values[i])
            hess_upper_y = gradient_values[i] + (diff * hessian_values[i])

            if gradient_values[i] < 0:
                direction = -1.
            else:
                direction = 1.

            ax_hess.plot([lower_x, upper_x],
                         [hess_lower_y, hess_upper_y],
                         'k-')


        dy = gradient_values * diff
        dx = np.ones_like(points) * diff
        dx[gradient_values < 0] *= -1.
        dy[gradient_values < 0] *= -1.

        ax_grad.quiver(points, llh_values, dx, dy, angles='xy', scale_units='xy', scale=1.)
        ax_grad.plot(best_fit[selected_bin], llh_values[selected_bin], 'ro')
        ax_hess.plot(points, gradient_values, 'o')
        fig.savefig('05_llh_scan.png')
        plt.close(fig)
        return selected_bin

    logging.info('Creating plot of a LLH slice')
    fig = plot_llh_slice(llh, best_fit.x)
    fig.savefig('05_llh_slice.png')

    logging.info('Creating corner plot')
    corner_fig = corner.corner(sample,
                               truths=vec_f_est_mcmc,
                               truth_color='r',
                               rasterized=True)
    corner_fig.savefig('05_corner_fact.png')

    logging.info('Creating best fit plots')
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bin_mids = (binning_E[1:] + binning_E[:-1]) / 2.
    bin_width = (binning_E[1:] - binning_E[:-1]) / 2.
    plt.hist(bin_mids, bins=binning_E, weights=vec_f_truth, histtype='step')
    ax.errorbar(bin_mids,
                best_fit.x,
                yerr=np.sqrt(np.diag(np.absolute(mini_cov))),
                xerr=bin_width,
                ls="",
                color="k",
                label="Unfolding (Minimizer)")
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylim([2e1, 2e3])
    fig.savefig('05_unfolding_minimizer.png')
    plt.close(fig)


    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.hist(bin_mids, bins=binning_E, weights=vec_f_truth, histtype='step')
    ax.errorbar(bin_mids,
                vec_f_est_mcmc,
                yerr=[vec_f_est_mcmc - sigma_vec_f[0, :],
                      sigma_vec_f[1, :] - vec_f_est_mcmc],
                xerr=bin_width,
                ls="",
                color="r",
                label="Unfolding (MCMC)")
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylim([2e1, 2e3])
    fig.savefig('05_unfolding_mcmc.png')
    plt.close(fig)

    logging.info('Creating LLH histogram')
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.hist(2*(np.max(probs) - probs),
            bins=50,
            weights=np.ones_like(probs) * 1./len(probs),
            histtype='step', lw=2)
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel(r'$-2\cdot\ln\left(\frac{\mathdefault{LLH}}{\mathdefault{LLH}_{\mathdefault{Best Fit}}}\right)$')
    ax.set_ylabel(r'$\frac{\mathdefault{Bin}_i}{\sum_i \mathdefault{Bin}_i}$')
    plt.tight_layout()
    plt.savefig('05_hist_probs.png')
