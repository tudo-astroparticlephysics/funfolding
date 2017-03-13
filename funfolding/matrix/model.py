import logging

class Model:
    name = 'BaseModel'

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1

    def initialize(self, X=None, y=None):
        self.logger.debug('Initilizing the model!')
        self.status = 0

    def eval(self, y=None):
        self.logger.debug('Model evaluation!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
