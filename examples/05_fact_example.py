import logging

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from funfolding import binning, model, solution
from funfolding.visualization.visualize_classic_binning import plot_binning
from funfolding.visualization.visualize_classic_binning import mark_bin

import corner
from scipy import linalg


def read_in(filename='Gamma_clas_sep.hdf5'):
    df = pd.read_hdf(filename)
    df_cutted = df[df.confidence_true_ >= 0.9]

    df_cutted.MCorsikaEvtHeader_fTotalEnergy = np.log10(
        df_cutted.MCorsikaEvtHeader_fTotalEnergy)
    df_cutted.E_RF = np.log10(df_cutted.E_RF)
    df_cutted.ConcCore = np.log10(df_cutted.ConcCore)
    df_cutted.Size = np.log10(df_cutted.Size)
    df_cutted.Length = np.log10(df_cutted.Length)
    df_cutted.numPixelInShower = np.log10(
    df_cutted.numPixelInShower)

    df_cutted = df_cutted[df_cutted.MCorsikaEvtHeader_fTotalEnergy <= 4.2]
    df_cutted = df_cutted[df_cutted.MCorsikaEvtHeader_fTotalEnergy >= 2.4]

    df_cutted = df_cutted[df_cutted.ZdTracking <= 31.0]
    df_cutted = df_cutted[df_cutted.ZdTracking >= 5]

    return df_cutted




if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

    random_state = np.random.RandomState(1337)

    df = read_in()
    df_A = df.iloc[5000:]
    df_test = df.iloc[:5000]

    X = df_A.get(['ConcCore', 'E_RF']).values
    X_test = df_test.get(['ConcCore', 'E_RF']).values

    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(df_A.MCorsikaEvtHeader_fTotalEnergy,
                           binning_E)
    binned_E_test = np.digitize(df_test.MCorsikaEvtHeader_fTotalEnergy,
                           binning_E)
    classic_binning = binning.ClassicBinning(
        bins=[15, 25],
        random_state=random_state)
    classic_binning.fit(X)

    fig, ax = plt.subplots()
    plot_binning(ax,
                 classic_binning,
                 X,
                 log_c=False,
                 cmap='viridis')
    fig.savefig('05_fact_example_original_binning.png')

    closest = classic_binning.merge(X_test,
                                    min_samples=10,
                                    max_bins=None,
                                    mode='closest')
    fig, ax = plt.subplots()
    plot_binning(ax,
                 closest,
                 X,
                 log_c=False,
                 cmap='viridis')
    fig.savefig('05_fact_example_original_binning_closest.png')

    unmerged_model = model.LinearModel()
    binned_g = classic_binning.digitize(X)
    unmerged_model.initialize(digitized_obs=binned_g,
                              digitized_truth=binned_E)


    merged_model = model.LinearModel()
    binned_g = closest.digitize(X)
    merged_model.initialize(digitized_obs=binned_g,
                     digitized_truth=binned_E)

    single_obs_model = model.LinearModel()
    max_e = np.max(X[:, 1]) + 1e-3
    min_e = np.min(X[:, 1]) - 1e-3
    binning_E_obs = np.linspace(min_e, max_e, 11)
    binned_g = np.digitize(X[:, 1], binning_E_obs)
    single_obs_model.initialize(digitized_obs=binned_g,
                                digitized_truth=binned_E)


    n_bins = len(closest.i_to_t)
    single_obs_model_more_bins = model.LinearModel()
    max_e = np.max(X[:, 1]) + 1e-3
    min_e = np.min(X[:, 1]) - 1e-3
    binning_E_obs = np.linspace(min_e, max_e, n_bins + 1)
    binned_g = np.digitize(X[:, 1], binning_E_obs)
    single_obs_model_more_bins.initialize(digitized_obs=binned_g,
                                          digitized_truth=binned_E)

    vec_g, vec_f = merged_model.generate_vectors(binned_g, binned_E)


    tree_obs = ["Size",
                "Width",
                "Length",
                "M3Trans",
                "M3Long",
                "ConcCore",
                "m3l",
                "m3t",
                "Concentration_onePixel",
                "Concentration_twoPixel",
                "Leakage",
                "Leakage2",
                "concCOG",
                "numIslands",
                "numPixelInShower",
                "phChargeShower_mean",
                "phChargeShower_variance",
                "phChargeShower_max"]

    X_tree = df_A.get(tree_obs).values
    X_tree_test = df_test.get(tree_obs).values



    tree_binning_uniform = binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=100,
        max_leaf_nodes=100,
        random_state=random_state)

    tree_binning_uniform.fit(X_tree,
                     binned_E,
                     uniform=True)


    binned_g = tree_binning_uniform.digitize(X_tree)

    tree_model_uniform = model.LinearModel()
    tree_model_uniform.initialize(digitized_obs=binned_g,
                          digitized_truth=binned_E)

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=100,
        max_leaf_nodes=100,
        random_state=random_state)

    tree_binning.fit(X_tree,
                     binned_E,
                     uniform=False)

    binned_g = tree_binning.digitize(X_tree)

    tree_model = model.LinearModel()
    tree_model.initialize(digitized_obs=binned_g,
                          digitized_truth=binned_E)

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

    svd_values = single_obs_model.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='Single Observable ({} Bins)'.format(closest.n_bins))

    svd_values = single_obs_model_more_bins.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='Single Observable ({} Bins)'.format(closest.n_bins))

    svd_values = tree_model.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='Tree Based ({} Bins)'.format(tree_binning.n_bins))

    svd_values = tree_model_uniform.evaluate_condition()
    ax.hist(bin_centers,
            bins=bin_edges,
            weights=svd_values,
            histtype='step',
            label='Tree Based ({} Bins; Uniform)'.format(tree_binning.n_bins))

    plt.legend(loc='lower left')
    ax.set_yscale("log", nonposy='clip')
    plt.savefig('05_condition.png')

    binned_g_test = tree_binning.digitize(X_tree_test)
    vec_g, vec_f = tree_model.generate_vectors(binned_g_test,
                                               binned_E_test)

    llh = solution.StandardLLH(tau=None,
                               C='thikonov')
    llh.initialize(vec_g=vec_g,
                    model=tree_model)


    sol_gd = solution.LLHSolutionGradientDescent(n_steps=500,
                                                 gamma=0.01)
    sol_gd.initialize(llh=llh, model=tree_model)
    sol_gd.set_x0_and_bounds()
    x, llh_values, gradient, hessian = sol_gd.fit()
    idx_best = np.argmax(llh_values)
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(x[idx_best], vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))
    covariance = linalg.inv(hessian[-1] * -1)
    print(np.sqrt(np.diag(covariance)))

    def create_llh_slice(llh, best_fit, selected_bin=None):
        if selected_bin is None:
            selected_bin = np.argmax(best_fit)
        points = np.linspace(0.9 * best_fit[selected_bin],
                             1.1 * best_fit[selected_bin],
                             101)
        llh_values = np.zeros_like(points)
        gradient_values = np.zeros_like(points)
        hessian_values = np.zeros_like(points)

        fig, [ax_grad, ax_hess] = plt.subplots(2, 1, figsize=(24, 18))
        diff = np.diff(points)[0] / 2.
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

            ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')

            ax_grad.plot([lower_x, upper_x],
                         [grad_lower_y, grad_upper_y],
                         'k-')

            ax_hess.plot([lower_x, upper_x],
                         [hess_lower_y, hess_upper_y],
                         'k-')

        ax_grad.plot(points, llh_values, 'o')
        ax_hess.plot(points, gradient_values, 'o')
        fig.savefig('05_llh_scan.png')
        plt.close(fig)
        return selected_bin

    create_llh_slice(llh, x[idx_best])

    print('\nMCMC Solution: (constrained: sum(vec_f) == sum(vec_g)) :')
    sol_mcmc = solution.LLHSolutionMCMC(n_burn_steps=100,
                                        n_used_steps=1000,
                                        n_walker=100,
                                        random_state=random_state)
    sol_mcmc.initialize(llh=llh, model=tree_model)
    sol_mcmc.set_x0_and_bounds(x0=x[idx_best])
    vec_f_est_mcmc, sigma_vec_f, sample, probs = sol_mcmc.fit()
    sol_mcmc.n_threads = 9
    print('Calculating Eff sample size')
    n_eff = sol_mcmc.calc_effective_sample_size(sample, n_threads=9)
    print(n_eff)
    std = np.std(sample, axis=0)
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mcmc, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))

    exit()

    corner_fig = corner.corner(sample,
                               #truths=vec_f_est_mcmc,
                               #truth_color='r',
                               rasterized=True)
    corner_fig.savefig('05_corner_fact.png')


    def add_path_to_corner_plot(corner_fig, points, step_size=1):
        used_points = points[::step_size]
        axes = corner_fig.axes

        for n in range(len(used_points)):
            lines = []
            pointer = -1
            points = used_points[:n+1]
            for i_y in range(points.shape[1]):
                for i_x in range(points.shape[1]):
                    pointer += 1
                    ax = axes[pointer]
                    if i_x > i_y:
                        continue
                    elif i_x == i_y:
                        x_lims = ax.get_xlim()
                        p_x = points[:, i_x]
                        x_in_range = np.logical_and(x_lims[0] < p_x,
                                                    x_lims[1] > p_x)
                        if sum(x_in_range) > 0:
                            p_x = points[x_in_range, i_x]
                            lines.append(ax.axvline(p_x[-1], color='b'))
                    else:
                        x_lims = ax.get_xlim()
                        y_lims = ax.get_ylim()
                        p_x = points[:, i_x]
                        p_y = points[:, i_y]
                        x_in_range = np.logical_and(x_lims[0] < p_x,
                                                    x_lims[1] > p_x)
                        y_in_range = np.logical_and(y_lims[0] < p_y,
                                                    y_lims[1] > p_y)
                        in_range = np.logical_and(x_in_range, y_in_range)
                        if sum(in_range) > 0:
                            p_x = points[in_range, i_x]
                            p_y = points[in_range, i_y]
                            lines.append(ax.plot(p_x, p_y, 'o-', color='b'))
            if len(lines) > 0:
                corner_fig.savefig(
                    'gif_jpgs/05_corner_numbers_{}.jpg'.format(n),
                    figsize=(8, 8),
                    dpi=50)
            for l in lines:
                try:
                    l.pop(0).remove()
                except AttributeError:
                    l.remove()


    add_path_to_corner_plot(corner_fig, x, step_size=10)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bin_mids = (binning_E[1:] + binning_E[:-1]) / 2.
    bin_width = (binning_E[1:] - binning_E[:-1]) / 2.
    _, vec_f_truth = tree_model.generate_vectors(binned_g,
                                                 binned_E)
    vec_f_truth = vec_f_truth * sum(vec_f) / sum(vec_f_truth)
    plt.hist(bin_mids, bins=binning_E, weights=vec_f_truth, histtype='step')
    ax.errorbar(bin_mids,
                vec_f_est_mcmc,
                yerr=std,
                xerr=bin_width,
                ls="",
                color="k",
                label="Unfolding (Error: Std)")
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylim([2e1, 2e3])
    fig.savefig('05_unfolding_mcmc_std.png')
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
                label="Unfolding (Error: Bayesian)")
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylim([2e1, 2e3])
    fig.savefig('05_unfolding_mcmc_bayesian.png')
    plt.close(fig)



    import cPickle

    with open('probs.dat', 'wb') as f:
        cPickle.dump(probs, f)

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
