import logging

import numpy as np
from matplotlib import pyplot as plt

from funfolding import binning
from funfolding.visualization.visualize_classic_binning import plot_binning

if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

    X = np.random.normal(size=20000)
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

    classic_binning = binning.ClassicBinning(
        bins = [25, 25],
        range=[[-5, 5], [-5, 5]])
    classic_binning.initialize(X)

    fig, ax = plt.subplots()
    plot_binning(
        ax,
        classic_binning,
        X,
        log_c=False,
        cmap='viridis')
    fig.savefig('00_example_gaussian_unmerged.png')

    threshold = 50

    closest = classic_binning.merge(X,
                                    min_samples=threshold,
                                    max_bins=None,
                                    mode='closest')
    binned = closest.histogram(X)
    fig, ax = plt.subplots()
    plot_binning(
        ax,
        closest,
        X,
        log_c=False,
        cmap='viridis')
    fig.savefig('00_example_gaussian_closest.png')

    lowest = classic_binning.merge(X,
                                   min_samples=threshold,
                                   max_bins=None,
                                   mode='lowest')
    binned = lowest.histogram(X)
    fig, ax = plt.subplots()
    plot_binning(ax,
                 lowest,
                 X,
                 log_c=False,
                 cmap='viridis')
    fig.savefig('00_example_gaussian_lowest.png')



    fig, ax = plt.subplots(1, 2, figsize=(24, 9))
    similar_clf = classic_binning.merge(X,
                                        min_samples=threshold,
                                        max_bins=None,
                                        mode='similar',
                                        y=y_clf)
    binned = similar_clf.histogram(X)
    plot_binning(ax[0],
                 similar_clf,
                 X,
                 log_c=False,
                 cmap='viridis')
    ax[1].hexbin(X[:, 0],
                 X[:, 1],
                 C=y_clf,
                 gridsize=50,
                 cmap='inferno')
    ax[1].set_xlim([-5, 5])
    ax[1].set_ylim([-5, 5])
    fig.savefig('00_example_gaussian_similar_clf.png')

    fig, ax = plt.subplots(1, 2, figsize=(24, 9))
    similar_reg = classic_binning.merge(X,
                                    min_samples=threshold,
                                    max_bins=None,
                                    mode='similar',
                                    y=y_reg)
    binned = similar_reg.histogram(X)
    plot_binning(ax[0],
                 similar_reg,
                 X,
                 log_c=False,
                 cmap='viridis')
    ax[1].hexbin(X[:, 0],
                 X[:, 1],
                 C=y_reg,
                 gridsize=50,
                 cmap='inferno')
    ax[1].set_xlim([-5, 5])
    ax[1].set_ylim([-5, 5])
    fig.savefig('00_example_gaussian_similar_reg.png')
