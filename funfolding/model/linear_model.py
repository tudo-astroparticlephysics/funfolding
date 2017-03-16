import logging
import numpy as np

class LinearModel:
    name = 'LinearModel'

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1
        self.A = None

    def initialize(self, X=None, y=None):
        self.logger.debug('Initilizing the model!')
        self.status = 0

    def evaluation(self, f=None):
        self.logger.debug('Model evaluation!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        return np.dot(self.A, f)

class BasicLinearModel(LinearModel):
    name = 'BasicLinearModel'

    def initialize(self, X, y, sample_weight=None):
        super(BasicLinearModel, self).initialize()
        X_binning = np.unique(X)
        X_binning = np.vstack((X_binning -0.5, X_binning[-1] + 0.5))
        y_binning = np.unique(y)
        y_binning = np.vstack((y_binning -0.5, y_binning[-1] + 0.5))
        self.A = np.histogram2d(x=X,
                                y=y,
                                bins=(X_binning, y_binning),
                                weights=sample_weight)[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = np.dot(self.A, M_norm)
