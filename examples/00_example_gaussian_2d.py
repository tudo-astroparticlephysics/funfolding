import logging

import numpy as np
from matplotlib import pyplot as plt

from funfolding import discretization

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

    classic_binning = discretization.ClassicBinning(
        bins = [25, 25],
        range=[[-5, 5], [-5, 5]])
    classic_binning.fit(X)

    fig, ax = plt.subplots()
    discretization.visualize_classic_binning(ax,
                                             classic_binning,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    fig.savefig('00_example_gaussian_unmerged.png')

    threshold = 20

    closest = classic_binning.merge(X,
                                    min_samples=threshold,
                                    max_bins=None,
                                    mode='closest')
    fig, ax = plt.subplots()
    discretization.visualize_classic_binning(ax,
                                             closest,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    fig.savefig('00_example_gaussian_closest.png')

    lowest = classic_binning.merge(X,
                                   min_samples=threshold,
                                   max_bins=None,
                                   mode='lowest')

    fig, ax = plt.subplots()
    discretization.visualize_classic_binning(ax,
                                             lowest,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    fig.savefig('00_example_gaussian_lowest.png')

    similar_clf = classic_binning.merge(X,
                                        min_samples=threshold,
                                        max_bins=None,
                                        mode='similar',
                                        y=y_clf)

    fig, ax = plt.subplots()
    discretization.visualize_classic_binning(ax,
                                             similar_clf,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    fig.savefig('00_example_gaussian_similar_clf.png')

    similar_reg = classic_binning.merge(X,
                                    min_samples=threshold,
                                    max_bins=None,
                                    mode='similar',
                                    y=y_reg)

    fig, ax = plt.subplots()
    discretization.visualize_classic_binning(ax,
                                             similar_reg,
                                             X,
                                             log_c=False,
                                             cmap='viridis')
    fig.savefig('00_example_gaussian_similar_reg.png')
