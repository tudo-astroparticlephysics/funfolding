import logging

import numpy as np


class Binning:
    name = 'Binning'
    status_need_for_digitize = 0

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.info('Created {}'.format(self.name))
        self.n_bins = None
        self.status = -1

    def initialize(self):
        self.logger.info("Building initial Binning!")
        self.status = 0

    def fit(self, *args, **kwargs):
        return self.initialize(*args, **kwargs)

    def digitize(self):
        if self.status < self.status_need_for_digitize:
            raise RuntimeError("Binning needs to be fitted! "
                               "Run 'Binning.initialize' first!")
        self.logger.info('Digitizing values!')

    def merge(self):
        self.logger.info('Merging bins of the model')
        if self.status == -1:
            raise RuntimeError("Run 'initialize' before 'reduce'!")
        elif self.status > 0:
            self.logger.warn("Binning is already merged {} times!".format(
                self.status))
        self.status += 1

    def prune(self, *args, **kwargs):
        return self.merge(*args, **kwargs)

    def histogram(self, X=None, sample_weight=None):
        self.logger.info('Building a histogram!')
        if sample_weight is not None:
            original_sum = np.sum(sample_weight)
        else:
            original_sum = X.shape[0]
        binned = self.digitize(X=X, sample_weight=sample_weight)
        counted = np.bincount(binned,
                              weights=sample_weight,
                              minlength=self.n_bins)
        assert np.sum(counted) == original_sum
        return counted
