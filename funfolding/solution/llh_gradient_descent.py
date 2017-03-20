import numpy as np
from scipy import linalg
from scipy.optimize import minimize

from .solution import Solution
from ..model import LinearModel



def get_tau(ndof, S_eig):
    if ndof >= len(S_eig):
        raise ValueError("'ndof' has to be < len(S_eig)")
    def f(tau):
        if tau < 0:
            return np.inf
        else:
            return np.absolute(np.sum(1 / (1 + tau * S_eig)) - ndof)
    tau_result = minimize(f, [1])
    if not tau_result.success:
        raise RuntimeError('Tau fit could not converge!')

    return tau_result.x[0]

