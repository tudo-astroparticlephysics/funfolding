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
        self.range_g = None
        self.range_f = None

    def __generate_binning__(self):
        binnings = []
        for r in [self.range_g, self.range_f]:
            low = r[0]
            high = r[-1]
            binnings.append(np.linspace(low, high + 1, high - low + 2))
        return binnings[0], binnings[1]

    def generate_vectors(self, g=None, f=None):
        binning_g, binning_f = self.__generate_binning__()
        if g is not None:
            vec_g = np.histogram(g, bins=binning_g)[0]
        else:
            vec_g = None
        if f is not None:
            vec_f = np.histogram(f, bins=binning_f)[0]
        else:
            vec_f = None
        return vec_g, vec_f

    def initialize(self, g, f, sample_weight=None):
        super(BasicLinearModel, self).initialize()
        self.range_g = (min(g), max(g))
        self.range_f = (min(f), max(f))
        binning_g, binning_f = self.__generate_binning__()

        self.A = np.histogram2d(x=g, y=f, bins=(binning_g, binning_f))[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=1))
        self.A = np.dot(M_norm, self.A)
