import logging
import numpy as np


class Solution(object):
    name = 'Solution'

    def __init__(self, random_state=None):
        super(Solution, self).__init__()
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1
        if not isinstance(random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

    def initialize(self):
        self.logger.debug('Initilizing the Solution!')
        self.status = 0

    def run(self, vec_g=None, model=None):
        self.logger.debug('Running Solution!')
        if self.status < 0:
            raise RuntimeError("Solution has to be intilized. "
                               "Run 'solver.initialize' first!")
