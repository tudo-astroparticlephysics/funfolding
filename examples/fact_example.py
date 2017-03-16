import logging

import pandas as pd
import numpy as np

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

    min_rf_e = np.min(df.E_RF)
    max_rf_e = np.max(df.E_RF)

    min_conc = np.min(df.ConcCore)
    max_conc = np.max(df.ConcCore)


    X = df.get(['ConcCore', 'E_RF']).values


    classic_binning = discretization.ClassicBinning(
        bins = [15, 25],
        range=[[min_conc, max_conc], [min_rf_e, max_rf_e]])
    classic_binning.fit(X)
    binned_g = classic_binning.digitize(X)
    vec_g = classic_binning.histogram(X)

    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(df.MCorsikaEvtHeader_fTotalEnergy,
                           binning_E)

    model = model.BasicLinearModel()
    model.initialize(X=binned_g,
                     y=binned_E)

    vec_f, V_f_est = solution.SVDSolution(vec_g=vec_g,
                                          model=model,
                                          keep_n_sig_values=5)


