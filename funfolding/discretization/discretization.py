import logging

class Discretization:
    name = 'BaseClassDiscretization'

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1

    def fit(self, X=None, y=None, sample_weights=None):
        self.logger.debug('Fitting Discretization')
        self.status = 0

    def digitize(self, X=None, sample_weights=None):
        self.logger.debug('Digitizing Data')

    def reduce(self, X=None):
        self.logger.debug('Reducing the model')
        if self.status == -1:
            raise RuntimeError("Run 'fit' before 'reduce'!")
        elif self.status == 1:
            self.logger.warn("Model is already reduced!")
        self.status = 1
