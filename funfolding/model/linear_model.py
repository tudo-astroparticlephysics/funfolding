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

    def evaluate(self, f=None):
        self.logger.debug('Model evaluation!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        return np.dot(self.A, f)


class BasicLinearModel(LinearModel):
    name = 'BasicLinearModel'
    def __init__(self):
        super(BasicLinearModel, self).__init__()
        self.range_x = None
        self.range_y = None

    def __generate_binning__(self):
        binnings = []
        for r in [self.range_x, self.range_y]:
            low = r[0]
            high = r[-1]
            binnings.append(np.linspace(low, high + 1, high - low + 2))
        return binnings[0], binnings[1]

    def generate_vectors(self, X=None, y=None):
        binning_x, binning_y = self.__generate_binning__()
        if X is not None:
            vec_X = np.histogram(X, bins=binning_x)[0]
        else:
            vec_X = None

        if y is not None:
            vec_y = np.histogram(y, bins=binning_y)[0]
        else:
            vec_y = None
        return vec_X, vec_y

    def initialize(self, X, y, sample_weight=None):
        super(BasicLinearModel, self).initialize()
        self.range_x = (min(X), max(X))
        self.range_y = (min(y), max(y))
        binning_x, binning_y = self.__generate_binning__()

        self.A = np.histogram2d(x=y, y=X, bins=(binning_y, binning_x))[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = np.dot(self.A, M_norm)
