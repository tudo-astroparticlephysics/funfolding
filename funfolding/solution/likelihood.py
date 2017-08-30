import warnings

import numpy as np

from ..model import LinearModel, Model


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


class LLH(object):
    name = 'LLH'
    status_need_for_eval = 0

    def __init__(self):
        self.status = -1

        self.vec_g = None
        self.model = None

        self.gradient_defined = False
        self.hesse_matrix_defined = False

    def initialize(self):
        self.status = 0

    def evaluate_llh(self):
        if self.status < 0:
            raise RuntimeError("LLH has to be intilized. "
                               "Run 'LLH.initialize' first!")

    def evaluate_gradient(self):
        if self.gradient_defined:
            if self.status < 0:
                raise RuntimeError("LLH has to be intilized. "
                                   "Run 'LLH.initialize' first!")
        else:
            raise NotImplementedError("Gradients are not implemented!")

    def evaluate_hesse_matrix(self):
        if self.hesse_matrix_defined:
            if self.status < 0:
                raise RuntimeError("LLH has to be intilized. "
                                   "Run 'LLH.initialize' first!")
        else:
            raise NotImplementedError("Hesse Matrix is not implemented!")

    def __call__(self, f):
        return self.evaluate_llh(f)


class StandardLLH(LLH):
    name = 'StandardLLH'
    status_need_for_eval = 0

    def __init__(self,
                 tau=None,
                 C='thikonov',
                 vec_acceptance=None,
                 log_f=False,
                 neg_llh=True):
        super(StandardLLH, self).__init__()
        self.C = C
        self.tau = tau
        if neg_llh:
            self.factor = 1.
        else:
            self.factor = -1.
        self.log_f_reg = log_f
        self.vec_acceptance = vec_acceptance

    def initialize(self,
                   vec_g,
                   model):
        super(StandardLLH, self).initialize()
        if not isinstance(model, Model):
            raise ValueError("'model' has to be of type Model!")
        self.model = model
        self.vec_g = vec_g
        self.N = np.sum(vec_g)

        if self.tau is None:
            self._tau = None
        else:
            if isinstance(self.tau, float):
                if self.tau <= 0.:
                    self._tau = None
                else:
                    self._tau = np.ones(model.dim_f) * self.tau
            elif callable(self.tau):
                self._tau = self.tau(np.arange(model.dim_f))
            else:
                raise ValueError("'tau' as to be either None, float or "
                                 "callable!")
            if self._tau is not None:
                if isinstance(self.C, str):
                    if self.C.lower() == 'thikonov' or self.C.lower() == '2':
                        m_C = create_C_thikonov(model.dim_f)
                elif isinstance(self.C, int):
                    if self.C == 2:
                        m_C = create_C_thikonov(model.dim_f)
                if m_C is None:
                    raise ValueError("{} invalid option for 'C'".format(
                        self.C))
                self._C = np.dot(np.dot(m_C, np.diag(self._tau)), m_C)

        if self.vec_acceptance is not None:
            if len(self.vec_acceptance) != model.dim_f:
                raise ValueError("'vec_acceptance' has to be of the same "
                                 "length as vec_f!")
            self.vec_acceptance = self.vec_acceptance
        else:
            self.vec_acceptance = 1.

        if isinstance(model, LinearModel):
            self.gradient_defined = True
            self.hesse_matrix_defined = True

    def evaluate_llh(self, f):
        super(StandardLLH, self).evaluate_llh()
        g_est, f, f_reg = self.model.evaluate(f)
        if any(g_est < 0) or any(f < 0):
            return np.inf * self.factor
        poisson_part = np.sum(g_est - self.vec_g * np.log(g_est))
        if self._tau is not None:
            f_reg_used = f_reg * self.vec_acceptance
            if self.log_f_reg:
                f_reg_used = np.log10(f_reg_used + 1)
            regularization_part = 0.5 * np.dot(
                np.dot(f_reg_used.T, self._C), f_reg_used)
        else:
            regularization_part = 0
        return (poisson_part - regularization_part) * self.factor

    def evaluate_gradient(self, f):
        super(StandardLLH, self).evaluate_gradient()
        warnings.warn('Gradient is not tested!')
        g_est, f, f_reg = self.model.evaluate(f)
        h_unreg = np.sum(self.model.A, axis=0)
        part_b = np.sum(self.model.A.T * self.vec_g * (1 / g_est), axis=1)
        h_unreg -= part_b
        regularization_part = np.ones_like(h_unreg) * self.tau * np.dot(
            self.C, f_reg)
        return h_unreg + regularization_part * self.factor

    def evaluate_hesse_matrix(self, f):
        raise NotImplementedError  # Bugged!
        super(StandardLLH, self).evaluate_hesse_matrix()
        warnings.warn('Gradient is not tested!')
        g_est, f, f_reg = self.model.evaluate(f)
        H_unreg = np.dot(np.dot(self.model.A.T,
                                np.diag(self.vec_g / g_est**2)),
                         self.model.A)
        return ((self.tau * self.C) + H_unreg) * self.factor


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

    def evaluate_llh(self, f):
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
