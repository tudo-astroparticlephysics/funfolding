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
