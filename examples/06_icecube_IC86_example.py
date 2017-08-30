import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from funfolding import binning, model, solution

import corner

if __name__ == '__main__':
    df = pd.read_hdf('sampled_2.hdf5')

    n_test = int(len(df) / 10)

    print(n_test)

    df_A = df.iloc[n_test:]
    df_test = df.iloc[:n_test]

    unfolding_cols = [
        'HitMultiplicityValues.n_hit_strings',
        'HitMultiplicityValues.n_hit_doms',
        'HitMultiplicityValues.n_hit_doms_one_pulse',
        'HitStatisticsValues.cog_z_sigma',
        'HitStatisticsValues.min_pulse_time',
        'HitStatisticsValues.max_pulse_time',
        'HitStatisticsValues.q_max_doms',
        'HitStatisticsValues.z_min',
        'HitStatisticsValues.z_max',
        'HitStatisticsValues.z_mean',
        'HitStatisticsValues.z_sigma',
        'HitStatisticsValues.z_travel',
        'HitStatisticsValues.cog_x',
        'HitStatisticsValues.cog_y',
        'HitStatisticsValues.cog_z',
        'MuEXAngular4_Sigma.value',
        'SPEFit2BayesianFitParams.logl',
        'SPEFit2BayesianFitParams.rlogl',
        'SPEFit2_TTFitParams.logl',
        'SPEFit2_TTFitParams.rlogl',
        'SplineMPE.z',
        'SplineMPECharacteristics.avg_dom_dist_q_tot_dom',
        'SplineMPECharacteristics.empty_hits_track_length',
        'SplineMPECharacteristics.track_hits_separation_length',
        'SplineMPECharacteristics.track_hits_distribution_smoothness',
        'SplineMPEDirectHitsC.dir_track_length',
        'SplineMPEDirectHitsC.dir_track_hit_distribution_smoothness',
        'SplineMPEDirectHitsC.n_dir_strings',
        'SplineMPEDirectHitsC.n_dir_doms',
        'SplineMPEDirectHitsC.n_early_strings',
        'SplineMPEDirectHitsC.n_early_doms',
        'SplineMPEDirectHitsC.n_late_strings',
        'SplineMPEDirectHitsC.n_late_doms',
        'SplineMPEFitParams.logl',
        'SplineMPEFitParams.rlogl',
        'SplineMPEMuEXDifferential.z',
        'SplineMPEMuEXDifferential.energy',
        'SplineMPEMuEXDifferential_r.value',
        'SplineMPETruncatedEnergy_SPICEMie_AllBINS_MuEres.value',
        'SplineMPETruncatedEnergy_SPICEMie_AllBINS_Muon.energy',
        'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_MuEres.value',
        'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon.energy']

    classic_cols = ['SplineMPEMuEXDifferential.energy',
                    'SplineMPEDirectHitsC.n_dir_doms',
                    'SplineMPEDirectHitsC.dir_track_length']

    X_A_clf = df_A.get(unfolding_cols)
    X_A_classic = df_A.get(classic_cols)
    y_A = df_A.get('MCPrimary1.energy')

    X_test_clf = df_test.get(unfolding_cols)
    X_teset_classic = df_A.get(classic_cols)
    y_test = df_test.get('MCPrimary1.energy')

    binning_E = np.linspace(2.0, 5.0, 11)

    binned_E_A = np.digitize(y_A, binning_E)
    binned_E_test = np.digitize(y_test, binning_E)

    tree_binning_uniform = binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=100,
        max_leaf_nodes=100,
        random_state=1337)

    tree_binning_uniform.fit(
        X_A_clf,
        binned_E_A,
        uniform=True)

    binned_g_test = tree_binning_uniform.digitize(X_test_clf)

    tree_binning_uniform_model = model.BasicLinearModel()
    tree_binning_uniform_model.initialize(
        g=binned_g_test,
        f=binned_E_test)

    boosted_binning_uniform = binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=100,
        max_leaf_nodes=100,
        random_state=1337,
        boosted='SAMME.R')

    boosted_binning_uniform.fit(
        X_A_clf,
        binned_E_A,
        uniform=True)

    binned_g_test = boosted_binning_uniform.digitize(X_test_clf)

    boosted_binning_uniform_model = model.BasicLinearModel()
    boosted_binning_uniform_model.initialize(
        g=binned_g_test,
        f=binned_E_test)

    print('ASASFA')

    ax_condition = tree_binning_uniform_model.evaluate_condition(
        label='Tree Binning (Uniform)')
    ax_condition = boosted_binning_uniform_model.evaluate_condition(
        ax=ax_condition,
        label='Boosted Binning (Uniform)')

    ax_condition.set_yscale("log", nonposy='clip')
    print('CONDICTION')
    plt.savefig('06_condition.png')

    binned_g_test = tree_binning.digitize(X_test)
    vec_g, vec_f = tree_model.generate_vectors(binned_g_test,
                                               binned_E_test)
    print(vec_f)
    print('\nMCMC Solution: (constrained: sum(vec_f) == sum(vec_g)) :')
    llh_mcmc = solution.LLHSolutionMCMC(n_used_steps=4000,
                                        random_state=1337)
    llh_mcmc.initialize(vec_g=vec_g, model=tree_model)
    tau = 0.00002
    vec_f_est_mcmc, sample, probs = llh_mcmc.run(tau=tau)
    llh_sol = solution.LLHSolutionMinimizer()
    llh_sol.initialize(vec_g=vec_g, model=tree_model, bounds=True)
    solution, V_f_est = llh_sol.run(tau=tau, x0=vec_f_est_mcmc)
    vec_f_est_mini = solution.x
    std = np.std(sample, axis=0)
    quantiles = np.percentile(sample, [16, 84], axis=0)
    str_0 = 'unregularized:'
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est_mcmc, vec_f):
        str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
    print('{}\t{}'.format(str_0, str_1))


    corner.corner(sample, truths=vec_f_est_mini, truth_color='r')
    plt.savefig('06_corner_icecube.png')

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bin_mids = (binning_E[1:] + binning_E[:-1]) / 2.
    bin_width = (binning_E[1:] - binning_E[:-1]) / 2.
    _, vec_f_truth = tree_model.generate_vectors(binned_g_A,
                                                 binned_E_A)
    vec_f_truth = vec_f_truth * sum(vec_f) / sum(vec_f_truth)
    plt.hist(bin_mids, bins=binning_E, weights=vec_f_truth, histtype='step')
    ax.errorbar(bin_mids,
                vec_f_est_mcmc,
                yerr=std,
                xerr=bin_width,
                ls="",
                color="r", label="Unfolding")
    print(solution)
    print(vec_f_est_mini)
    print(quantiles[0, :])
    print(quantiles[1, :])
    quantiles[0, :] = vec_f_est_mini -quantiles[0, :]
    quantiles[1, :] = quantiles[1, :] - vec_f_est_mini
    print(quantiles)
    ax.errorbar(bin_mids,
                vec_f_est_mini,
                yerr=quantiles,
                xerr=bin_width,
                ls="",
                color="g", label="Hybrid")
    plt.legend(loc='best')
    ax.set_yscale("log", nonposy='clip')
    fig.savefig('06_unfolding_mcmc.png')


