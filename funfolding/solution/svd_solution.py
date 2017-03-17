import numpy as np
from scipy import linalg

from .solution import Solution
from ..model import LinearModel

class SVDSolution(Solution):
    name = 'SVDSolution'

    def run(self, vec_g, model, keep_n_sig_values):
        super(SVDSolution, self).run()
        if not isinstance(model, LinearModel):
            raise ValueError("'model' has to be of type LinearModel!")
        A = model.A
        m = A.shape[0]
        n = A.shape[1]
        U, S_values, V = linalg.svd(A)
        n_sig_values = int(min(len(S_values), keep_n_sig_values))
        order = np.argsort(S_values)[::-1]
        S_inv = np.zeros((n, m))
        for i in order[:n_sig_values]:
            S_inv[i, i] = 1. / np.real(S_values[i])
        A_inv = np.dot(V.T, np.dot(S_inv, U.T))
        vec_f = np.dot(A_inv, vec_g)
        vec_f = np.real(vec_f)
        V_y = np.diag(vec_g)
        V_f_est = np.real(np.dot(A_inv, np.dot(V_y, A_inv.T)))
        factor = np.sum(vec_g) / np.sum(vec_f)
        vec_f *= factor
        V_f_est *= factor
        return vec_f, V_f_est
