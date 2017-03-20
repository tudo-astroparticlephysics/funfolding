import numpy as np


from .solution import Solution
from ..model import LinearModel

class LLHMinimizerSolution(Solution):
    name = 'LLHMinimizerSolution'
    def __init__(self):
        super(LLHMinimizerSolution, self).__init__()
        self.original_singular_values = None
        self.used_singular_values = None
