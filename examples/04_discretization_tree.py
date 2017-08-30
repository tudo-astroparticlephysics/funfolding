import pandas as pd
import numpy as np

from funfolding import binning, model, solution


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
    df = read_in()
    df_A = df.iloc[5000:]
    df_test = df.iloc[:5000]

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
    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(df_A.MCorsikaEvtHeader_fTotalEnergy,
                           binning_E)
    binned_E_test = np.digitize(df_test.MCorsikaEvtHeader_fTotalEnergy,
                           binning_E)

    tree_binning = binning.TreeBinningSklearn()
    tree_binning.fit(X_tree_test, binned_E_test)


    print(tree_binning.tree.tree_.feature)
