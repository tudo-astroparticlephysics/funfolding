import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from funfolding import discretization, model, solution


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
    classic_binning = discretization.ClassicBinning(
        bins = [15, 25])
    classic_binning.fit(X)

    fig, ax = plt.subplots()
    discretization.visualize_classic_binning(ax,
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
    discretization.visualize_classic_binning(ax,
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
    single_obs_model_100 = model.BasicLinearModel()
    max_e = np.max(X[:, 1]) + 1e-3
    min_e = np.min(X[:, 1]) - 1e-3
    binning = np.linspace(min_e, max_e, n_bins + 1)
    binned_g = np.digitize(X[:, 1], binning)
    single_obs_model_100.initialize(g=binned_g,
                                    f=binned_E)

    vec_g, vec_f = merged_model.generate_vectors(binned_g, binned_E)
    ax_condition = unmerged_model.evaluate_condition(label='Unmerged')
    merged_model.evaluate_condition(ax=ax_condition, label='Merged')
    single_obs_model.evaluate_condition(ax=ax_condition,
                                        label='Single Observable (10 Bins)')
    single_obs_model_100.evaluate_condition(
        ax=ax_condition,
        label='Single Observable ({} Bins)'.format(n_bins))
    plt.legend(loc='lower left')
    ax_condition.set_yscale("log", nonposy='clip')
    plt.savefig('05_condition.png')

    exit()

    binned_g_unmerged_test = classic_binning.digitize(X_test)
    vec_g, vec_f = unmerged_model.generate_vectors(binned_g_unmerged_test,
                                                 binned_E_test)

    svd = solution.SVDSolution()
    print('\n===========================\nResults for each Bin: Unfolded/True')
    print('\nSVD Solution for diffrent number of kept sigular values:')
    for i in range(1, 10):
        vec_f_est, V_f_est = svd.run(vec_g=vec_g,
                                     model=unmerged_model,
                                     keep_n_sig_values=i)
        str_0 = '{} singular values:'.format(str(i).zfill(2))
        str_1 = ''
        for f_i_est, f_i in zip(vec_f_est, vec_f):
            str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
        print('{}\t{}'.format(str_0, str_1))

    binned_g_merged_test = closest.digitize(X_test)
    vec_g, vec_f = merged_model.generate_vectors(binned_g_merged_test,
                                                 binned_E_test)
    svd = solution.SVDSolution()
    print('\n===========================\nResults for each Bin: Unfolded/True')
    print('\nSVD Solution for diffrent number of kept sigular values:')
    for i in range(1, 10):
        vec_f_est, V_f_est = svd.run(vec_g=vec_g,
                                     model=merged_model,
                                     keep_n_sig_values=i)
        str_0 = '{} singular values:'.format(str(i).zfill(2))
        str_1 = ''
        for f_i_est, f_i in zip(vec_f_est, vec_f):
            str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
        print('{}\t{}'.format(str_0, str_1))


