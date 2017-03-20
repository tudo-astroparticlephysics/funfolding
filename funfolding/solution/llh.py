import emcee
import numpy as np
from scipy.optimize import minimize

from .solution import Solution
from ..model import LinearModel

def create_C_thikonov(n_dims):
    C = np.zeros((n_dims, n_dims))
    C[0, 0] = -1
    C[0, 1] = 1
    idx_N = n_dims - 1
    for i in range(1, idx_N):
        C[i, i] = -2.
        C[i, i-1] = 1
        C[i, i+1] = 1
    C[idx_N, idx_N] = -1
    C[idx_N, idx_N-1] = 1
    return np.dot(C.T, C)


class LLHThikonov:
    def __init__(self):
        self.g = None
        self.C = None
        self.n_dims_f = None
        self.tau = None
        self.linear_model = None
        self.status = -1

    def initialize(self, g, linear_model, tau):
        if not isinstance(lin_model, LinearModel):
            raise ValueError("'model' has to be of type LinearModel!")
        self.linear_model = linear_model
        self.n_dims_f = linear_model.A.shape[1]
        self.g = g
        self.C = create_second_derivative_matrix(self.n_dims_f)
        self.tau = tau
        self.status = 0

    def evaluate(self, f):
        g_est = self.linear_model.evaluation(f)
        poisson_part = np.sum(g * np.log(g_est) - g_est)
        regularization_part = 0.5*tau*np.dot(np.dot(f.T, self.C), f)
        neg_log_LLH = regularization_part - poisson_part
        return neg_log_LLH

    def __generate_gradient__(self, f, g_est):
        h_unreg = np.sum(self.linear_model.A, axis=1)
        part_b = np.dot(g_est, self.linear_model.A)
        part_b /= g_est
        h_unreg -= part_b
        reg_part = np.ones_like(h_unreg) * self.tau * np.dot(self.C, f)
        return h_unreg + reg_part

    def __generate_hesse_matrix__(self, f, g_est):
        H_unreg = np.dot(g_est,
                         np.dot(self.linear_model.A.T, self.linear_model.A))
        H_unreg /= g_est**2
        return H_unreg + self.tau * self.C






class LLHSolutionMinimizer(Solution):
    name = 'LLHSolutionMinimizer'
    def run(self, vec_g, model, tau, f_0):
        super(LLHSolutionMinimizer, self).run()
        LLH = LLHThikonov()
        LLH.initialize(g=vec_g, model=model, tau=tau)
        solution = minimize(fun=LLH.evaluate,
                            x_0=f_0)


class LLHSolutionGradientDescent(Solution):
    name = 'LLHSolutionBlobel'
    def run(self, vec_g, model, tau, f_0):
        super(LLHSolutionBlobel, self).run()
        LLH = LLHThikonov()
        LLH.initialize(g=vec_g, model=model, tau=tau)
        solution = minimize(fun=LLH.evaluate,
                            x_0=f_0)




if __name__ == '__main__':
    n_dims = 7
    C = create_second_derivative_matrix(n_dims)
    f = np.arange(n_dims)
    print(f)
    print(np.dot(C, f))

