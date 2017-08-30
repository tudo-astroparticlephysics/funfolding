import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from funfolding import binning, model, solution
from funfolding.visualization.visualize_classic_binning import plot_binning
from funfolding.visualization.visualize_classic_binning import mark_bin

import corner

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
        bins = [15, 25])
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

    unmerged_model = model.BasicLinearModel()
    binned_g = classic_binning.digitize(X)
    unmerged_model.initialize(g=binned_g,
                              f=binned_E)


    merged_model = model.BasicLinearModel()
    binned_g = closest.digitize(X)
    merged_model.initialize(g=binned_g,
                     f=binned_E)

    single_obs_model = model.BasicLinearModel()
    max_e = np.max(X[:, 1]) + 1e-3
    min_e = np.min(X[:, 1]) - 1e-3
    binning = np.linspace(min_e, max_e, 11)
    binned_g = np.digitize(X[:, 1], binning)
    single_obs_model.initialize(g=binned_g,
                                f=binned_E)


    n_bins = len(closest.i_to_t)
    single_obs_model_more_bins = model.BasicLinearModel()
    max_e = np.max(X[:, 1]) + 1e-3
    min_e = np.min(X[:, 1]) - 1e-3
    binning = np.linspace(min_e, max_e, n_bins + 1)
    binned_g = np.digitize(X[:, 1], binning)
    single_obs_model_more_bins.initialize(g=binned_g,
                                    f=binned_E)

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

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=100,
        max_leaf_nodes=100,
        random_state=1337)

    tree_binning.fit(X_tree,
                     binned_E,
                     uniform=True)


    binned_g = tree_binning.digitize(X_tree)

    tree_model = model.BasicLinearModel()
    tree_model.initialize(g=binned_g,
                          f=binned_E)

    ax_condition = unmerged_model.evaluate_condition(
        label='2 Observables (Unmerged; {} Bins)'.format(
            classic_binning.n_bins))
    merged_model.evaluate_condition(
        ax=ax_condition,
        label='2 Observables (Merged; {} Bins)'.format(closest.n_bins))
    single_obs_model.evaluate_condition(
        ax=ax_condition,
        label='Single Observable (10 Bins)')
    single_obs_model_more_bins.evaluate_condition(
        ax=ax_condition,
        label='Single Observable ({} Bins)'.format(closest.n_bins))
    tree_model.evaluate_condition(
        ax=ax_condition,
        label='Tree Based ({} Bins)'.format(tree_binning.n_bins))

    plt.legend(loc='lower left')
    ax_condition.set_yscale("log", nonposy='clip')
    plt.savefig('05_condition.png')

    binned_g_test = tree_binning.digitize(X_tree_test)
    vec_g, vec_f = tree_model.generate_vectors(binned_g_test,
                                               binned_E_test)
    print('\nMCMC Solution: (constrained: sum(vec_f) == sum(vec_g)) :')
    llh_mcmc = solution.LLHSolutionMCMC(n_used_steps=2000,
                                        random_state=1337)
    llh_mcmc.initialize(vec_g=vec_g, model=tree_model)
    vec_f_est_mcmc, sample, probs = llh_mcmc.run(tau=0)
    std = np.std(sample, axis=0)
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mcmc, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))

    # corner.corner(sample, truths=vec_f_est_mcmc, truth_color='r')
    # plt.savefig('05_corner_fact.png')

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
                color="r", label="Unfolding")
    ax.set_yscale("log", nonposy='clip')
    fig.savefig('05_unfolding_mcmc.png')
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
