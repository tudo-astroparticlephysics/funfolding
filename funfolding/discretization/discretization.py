import logging

class Discretization:
    name = 'BaseClassDiscretization'

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.n_bins = None
        self.status = -1

    def fit(self, X=None, y=None, sample_weight=None):
        self.logger.debug('Fitting Discretization')
        self.status = 0

    def digitize(self, X=None, sample_weight=None):
        self.logger.debug('Digitizing Data')

    def histogram(self, X=None, sample_weight=None):
        self.loggerl.debug('Building a histogram')
        if self.n_bins is None:
            raise RuntimeError("Numbers of bins unkown. Run 'fit' first!")
        binned = self.digitize(X=X, sample_weight=sample_weight)
        return np.bincount(binned,
                           weights=sample_weight,
                           minlength=self.n_bins)

    def merge(self, X=None):
        self.logger.debug('Reducing the model')
        if self.status == -1:
            raise RuntimeError("Run 'fit' before 'reduce'!")
        elif self.status == 1:
            self.logger.warn("Model is already reduced {} times!".format(
                self.status))
        self.status += 1
