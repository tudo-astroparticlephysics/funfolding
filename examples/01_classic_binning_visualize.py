import logging

import numpy as np
from matplotlib import pyplot as plt

from funfolding import binning
from funfolding.visualization.visualize_classic_binning import plot_binning
from funfolding.visualization.visualize_classic_binning import mark_bin


if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

    X = np.random.uniform(-5.5,  5.5,  size=30000)
    X = X.reshape((15000, 2))

    y_mean = 1.5
    y_clf = np.zeros(X.shape[0])
    is_pos = X[: ,0] * X[:, 1] > 0
    y_clf[is_pos] = 0
    y_clf[~is_pos] = 1
    y_means = np.sqrt(X[:, 0]**2 + X[:, 0]**2)
    y_reg = np.random.normal(loc=y_means,
                             scale=0.3)

    classic_binning = binning.ClassicBinning(
        bins = [20, 20],
        range=[[-5, 5], [-5, 5]])
    classic_binning.fit(X)
    hist = classic_binning.histogram(X)

    fig, ax = plt.subplots()
    plot_binning(ax,
                 classic_binning,
                 X,
                 log_c=False,
                 cmap='viridis')
    selected_bin = np.random.choice(list(classic_binning.i_to_t.keys()))
    neighbors = classic_binning.__get_neighbors__(selected_bin)
    for i in neighbors:
        mark_bin(ax,
                 classic_binning,
                 i,
                 color='r')
    mark_bin(ax,
             classic_binning,
             selected_bin,
             color='w')
    fig.savefig('01_marked_bins.png')


    fig, ax = plt.subplots()
    lowest = classic_binning.merge(X,
                                    min_samples=100,
                                    max_bins=None,
                                    mode='lowest')
    plot_binning(ax,
                 lowest,
                 X,
                 log_c=False,
                 cmap='viridis')
    selected_bin = np.random.choice(list(lowest.i_to_t.keys()))
    neighbors = lowest.__get_neighbors__(selected_bin)
    for i in neighbors:
        mark_bin(ax,
                                lowest,
                                i,
                                color='r')
    mark_bin(ax,
                            lowest,
                            selected_bin,
                            color='w')

    fig.savefig('01_marked_bins_merged_lowest.png')


    fig, ax = plt.subplots()
    closest = classic_binning.merge(X,
                                    min_samples=100,
                                    max_bins=None,
                                    mode='closest')
    plot_binning(ax,
                 closest,
                 X,
                 log_c=False,
                 cmap='viridis')
    selected_bin = np.random.choice(list(closest.i_to_t.keys()))
    neighbors = closest.__get_neighbors__(selected_bin)
    for i in neighbors:
        mark_bin(ax,
                 closest,
                 i,
                 color='r')
    mark_bin(ax,
             closest,
             selected_bin,
             color='w')

    fig.savefig('01_marked_bins_merged_closest.png')

    fig, ax = plt.subplots(1, 2, figsize=(24, 9))
    similar = classic_binning.merge(X,
                                    min_samples=100,
                                    max_bins=None,
                                    mode='similar',
                                    y=y_reg)
    plot_binning(ax[0],
                 similar,
                 X,
                 log_c=False,
                 cmap='viridis')
    selected_bin = np.random.choice(list(similar.i_to_t.keys()))
    neighbors = similar.__get_neighbors__(selected_bin)
    for i in neighbors:
        mark_bin(ax[0],
                 similar,
                 i,
                 color='r')
    mark_bin(ax[0],
             similar,
             selected_bin,
             color='w')
    ax[1].hexbin(X[:, 0],
                 X[:, 1],
                 C=y_reg,
                 gridsize=50,
                 cmap='inferno')
    ax[1].set_xlim([-5, 5])
    ax[1].set_ylim([-5, 5])
    fig.savefig('01_marked_bins_merged_similar_reg.png')

    fig, ax = plt.subplots(1, 2, figsize=(24, 9))
    similar = classic_binning.merge(X,
                                    min_samples=100,
                                    max_bins=None,
                                    mode='similar',
                                    y=y_clf)
    plot_binning(ax[0],
                 similar,
                 X,
                 log_c=False,
                 cmap='viridis')
    selected_bin = np.random.choice(list(similar.i_to_t.keys()))
    neighbors = similar.__get_neighbors__(selected_bin)
    for i in neighbors:
        mark_bin(ax[0],
                 similar,
                 i,
                 color='r')
    mark_bin(ax[0],
             similar,
             selected_bin,
             color='w')
    ax[1].hexbin(X[:, 0],
                 X[:, 1],
                 C=y_clf,
                 gridsize=50,
                 cmap='inferno')
    ax[1].set_xlim([-5, 5])
    ax[1].set_ylim([-5, 5])
    fig.savefig('01_marked_bins_merged_similar_clf.png')

