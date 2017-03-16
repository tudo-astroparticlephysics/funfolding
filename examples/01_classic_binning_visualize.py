import logging

import numpy as np
from matplotlib import pyplot as plt

from funfolding import discretization

if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

    X = np.random.uniform(-5.625,  5.625,  size=20000)
    X = X.reshape((10000, 2))

    y_mean = 1.5
    y_clf = np.zeros(X.shape[0])
    is_pos = X[: ,0] * X[:, 1] > 0
    y_clf[is_pos] = np.random.normal(loc=y_mean, size=np.sum(is_pos))
    y_clf[~is_pos] = np.random.normal(loc=-y_mean, size=np.sum(~is_pos))
    y_clf = np.array(y_clf >= 0, dtype=int)
    y_means = np.sqrt(X[:, 0]**2 + X[:, 0]**2)
    y_reg = np.random.normal(loc=y_means,
                             scale=0.3)

    classic_binning = discretization.ClassicBinning(
        bins = [15, 15],
        range=[[-5, 5], [-5, 5]])
    classic_binning.fit(X)
    hist = classic_binning.histogram(X)

    fig, ax = plt.subplots()
    discretization.visualize_classic_binning(ax,
                                             classic_binning,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    selected_bin = np.random.choice(list(classic_binning.i_to_t.keys()))
    neighbors = classic_binning.__get_neighbors__(selected_bin)
    for i in neighbors:
        discretization.mark_bin(ax,
                                classic_binning,
                                i,
                                color='r')
    discretization.mark_bin(ax,
                            classic_binning,
                            selected_bin,
                            color='w')

    fig.savefig('01_marked_bins.png')

    fig, ax = plt.subplots()
    closest = classic_binning.merge(X,
                                    min_samples=100,
                                    max_bins=None,
                                    mode='closest')
    discretization.visualize_classic_binning(ax,
                                             closest,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    selected_bin = np.random.choice(list(closest.i_to_t.keys()))
    neighbors = closest.__get_neighbors__(selected_bin)
    for i in neighbors:
        discretization.mark_bin(ax,
                                closest,
                                i,
                                color='r')
    discretization.mark_bin(ax,
                            closest,
                            selected_bin,
                            color='w')

    fig.savefig('01_marked_bins_merged.png')

    fig, ax = plt.subplots()
    similar = classic_binning.merge(X,
                                    min_samples=100,
                                    max_bins=None,
                                    mode='similar',
                                    y=y_clf)
    discretization.visualize_classic_binning(ax,
                                             similar,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    selected_bin = np.random.choice(list(similar.i_to_t.keys()))
    neighbors = similar.__get_neighbors__(selected_bin)
    for i in neighbors:
        discretization.mark_bin(ax,
                                similar,
                                i,
                                color='r')
    discretization.mark_bin(ax,
                            similar,
                            selected_bin,
                            color='w')
    fig.savefig('01_marked_bins_merged_similar.png')
