import logging


class Solution(object):
    name = 'Solution'

    def __init__(self):
        super(Solution, self).__init__()
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1

    def initialize(self):
        self.logger.debug('Initilizing the Solution!')
        self.status = 0

    def run(self, vec_g=None, model=None):
        self.logger.debug('Running Solution!')
        if self.status < 0:
            raise RuntimeError("Solution has to be intilized. "
                               "Run 'solver.initialize' first!")
