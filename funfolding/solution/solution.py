import logging

class Solution:
    name = 'Solution'

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1

    def initialize(self):
        self.logger.debug('Initilizing the Solution!')
        self.status = 0

    def run(self, g=None, model=None):
        self.logger.debug('Running Solution!')
        if self.status < 0:
            raise RuntimeError("Solution has to be intilized. "
                               "Run 'solver.initialize' first!")
