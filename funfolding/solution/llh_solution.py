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
        C[i, i - 1] = 1
        C[i, i + 1] = 1
    C[idx_N, idx_N] = -1
    C[idx_N, idx_N - 1] = 1
    return np.dot(C.T, C)


class LLHThikonov:
    def __init__(self, g, linear_model, tau):
        if not isinstance(linear_model, LinearModel):
            raise ValueError("'model' has to be of type LinearModel!")
        self.linear_model = linear_model
        self.n_dims_f = linear_model.A.shape[1]
        self.g = g
        self.C = create_C_thikonov(self.n_dims_f)
        self.tau = tau
        self.status = 0

    def evaluate_neg_log_llh(self, f):
        g_est, f = self.linear_model.evaluate(f)
        poisson_part = np.sum(g_est + self.g * np.log(g_est))
        regularization_part = 0.5 * self.tau * np.dot(np.dot(f.T, self.C), f)
        return regularization_part - poisson_part

    def evaluate_gradient(self, f):
        g_est, f = self.linear_model.evaluate(f)
        h_unreg = np.sum(self.linear_model.A, axis=0)
        part_b = np.sum(self.linear_model.A.T * self.g * (1 / g_est), axis=1)
        h_unreg -= part_b
        reg_part = np.ones_like(h_unreg) * self.tau * np.dot(self.C, f)
        return reg_part - h_unreg

    def evaluate_hesse_matrix(self, f):
        g_est, f = self.linear_model.evaluate(f)
        H_unreg = np.dot(np.dot(self.linear_model.A.T,
                                np.diag(self.g / g_est**2)),
                         self.linear_model.A)
        return self.tau * self.C + H_unreg


class LLHThikonovForLoops:
    def __init__(self, g, linear_model, tau):
        if not isinstance(linear_model, LinearModel):
            raise ValueError("'model' has to be of type LinearModel!")
        self.linear_model = linear_model
        self.n_dims_f = linear_model.A.shape[1]
        self.g = g
        self.C = create_C_thikonov(self.n_dims_f)
        self.tau = tau
        self.status = 0

    def evaluate_neg_log_llh(self, f):
        m, n = self.linear_model.A.shape
        poisson_part = 0
        for i in range(m):
            g_est = 0
            for j in range(n):
                g_est += self.linear_model.A[i, j] * f[j]
            poisson_part += g_est - self.g[i] * np.log(g_est)

        reg_part = 0
        for i in range(n):
            for j in range(n):
                reg_part += self.C[i, j] * f[i] * f[j]
        reg_part *= 0.5 * self.tau
        return reg_part - poisson_part

    def evaluate_gradient(self, f):
        m, n = self.linear_model.A.shape
        gradient = np.zeros(n)
        for k in range(n):
            poisson_part = 0
            for i in range(m):
                g_est = 0
                for j in range(n):
                    g_est += self.linear_model.A[i, j] * f[j]
                A_ik = self.linear_model.A[i, k]
                poisson_part += A_ik - (self.g[i] * A_ik) / g_est
            c = 0
            for i in range(n):
                c += self.C[i, k] * f[i]
            reg_part = self.tau * c
            gradient[k] = reg_part - poisson_part
        return gradient

    def evaluate_hesse_matrix(self, f):
        m, n = self.linear_model.A.shape
        hess = np.zeros((n, n))
        for k in range(n):
            for l in range(n):
                poisson_part = 0
                for i in range(m):
                    A_ik = self.linear_model.A[i, k]
                    A_il = self.linear_model.A[i, l]
                    nominator = self.g[i] * A_ik * A_il
                    denominator = 0
                    for j in range(n):
                        denominator += self.linear_model.A[i, j] * f[j]
                    poisson_part += nominator / denominator**2
                hess[k, l] = poisson_part + self.tau * self.C[k, l]
        return hess


class LLHSolutionMinimizer(Solution):
    name = 'LLHSolutionMinimizer'

    def run(self, vec_g, model, tau, x0=None, bounds=None):
        self.initialize()
        super(LLHSolutionMinimizer, self).run()
        if bounds is True:
            bounds = model.generate_bounds(vec_g)
        if x0 is None:
            x0 = model.generate_x0(vec_g)
        x0 = model.set_x0(x0)
        LLH = LLHThikonov(g=vec_g, linear_model=model, tau=tau)
        solution = minimize(fun=LLH.evaluate_neg_log_llh,
                            x0=x0,
                            bounds=bounds,
                            #  method='dogleg',
                            #  jac=LLH.evaluate_gradient,
                            #  hess=LLH.evaluate_hesse_matrix
                            )
        return solution
