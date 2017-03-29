import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import linalg

import emcee

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
    def __init__(self, g, linear_model, tau=0., N_prior=False, neg_llh=True):
        if not isinstance(linear_model, LinearModel):
            raise ValueError("'model' has to be of type LinearModel!")
        self.linear_model = linear_model
        self.n_dims_f = linear_model.A.shape[1]
        self.g = g
        self.C = create_C_thikonov(self.n_dims_f)
        self.tau = tau
        self.status = 0
        self.neg_llh = neg_llh

    def evaluate_llh(self, f):
        g_est, f = self.linear_model.evaluate(f)
        if any(g_est < 0) or any(f < 0):
            if self.neg_llh:
                return np.inf
            else:
                return -np.inf
        poisson_part = np.sum(g_est - self.g * np.log(g_est))
        regularization_part = 0.5 * self.tau * np.dot(np.dot(f.T, self.C), f)
        if self.neg_llh:
            return poisson_part + regularization_part
        else:
            return (poisson_part + regularization_part) * (-1)

    def evaluate_gradient(self, f):
        g_est, f = self.linear_model.evaluate(f)
        h_unreg = np.sum(self.linear_model.A, axis=0)
        part_b = np.sum(self.linear_model.A.T * self.g * (1 / g_est), axis=1)
        h_unreg -= part_b
        reg_part = np.ones_like(h_unreg) * self.tau * np.dot(self.C, f)
        if self.neg_llh:
            return poisson_part + regularization_part
        else:
            return (poisson_part + regularization_part) * (-1)

    def evaluate_hesse_matrix(self, f):
        g_est, f = self.linear_model.evaluate(f)
        H_unreg = np.dot(np.dot(self.linear_model.A.T,
                                np.diag(self.g / g_est**2)),
                         self.linear_model.A)
        if self.neg_llh:
            return (self.tau * self.C) + H_unreg
        else:
            return ((self.tau * self.C) + H_unreg) * (-1)


    def __call__(self, f):
        return self.evaluate_llh(f)


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


class LLHSolutionMinimizer(Solution):
    name = 'LLHSolutionMinimizer'

    def __init__(self):
        super(LLHSolutionMinimizer, self).__init__()
        self.llh = None
        self.vec_g = None
        self.bounds = None
        self.model = None

    def initialize(self, vec_g, model, bounds=None):
        super(LLHSolutionMinimizer, self).initialize()
        self.llh = LLHThikonov(g=vec_g, linear_model=model)
        self.vec_g = vec_g
        self.model = model
        if bounds is True:
           self.bounds = model.generate_bounds(vec_g)
        else:
            self.bounds = None

    def run(self, tau=None, x0=None):
        super(LLHSolutionMinimizer, self).run()
        if x0 is None:
            x0 = self.model.generate_x0(self.vec_g)
        x0 = self.model.set_x0(x0)
        if tau is not None and isinstance(tau, float):
            self.llh.tau = tau
        solution, V_f_est = self.__run_minimization__(x0)
        return solution.x, V_f_est

    def __run_minimization__(self, x0):
        cons = ({'type': 'eq', 'fun': lambda x: np.absolute(np.sum(x) -
                                                            np.sum(x0))})
        solution =  minimize(fun=self.llh.evaluate_llh,
                             x0=x0,
                             bounds=self.bounds,
                             method='SLSQP',
                             # jac=self.llh.evaluate_gradient,
                             # hess=self.llh.evaluate_hesse_matrix
                             constraints=cons)
        hess_matrix = self.llh.evaluate_hesse_matrix(solution.x)
        V_f_est = linalg.inv(hess_matrix)
        return solution, V_f_est


class LLHSolutionMCMC(Solution):
    name = 'LLHSolutionMCMC'
    def __init__(self,
                 n_walker=100,
                 n_used_steps=2000,
                 n_burn_steps=1000,
                 random_state=None):
        super(LLHSolutionMCMC, self).__init__(random_state=random_state)
        self.n_walker = n_walker
        self.n_used_steps = n_used_steps
        self.n_burn_steps = n_burn_steps
        self.llh = None
        self.vec_g = None
        self.model = None
        self.n_dims_f = None

    def initialize(self, vec_g, model):
        super(LLHSolutionMCMC, self).initialize()
        self.llh = LLHThikonov(g=vec_g, linear_model=model, neg_llh=False)
        self.n_dims_f = model.A.shape[1]
        self.vec_g = vec_g
        self.model = model

    def run(self, tau=None, x0=None):
        super(LLHSolutionMCMC, self).run()
        if x0 is None:
            x0 = self.model.generate_x0(self.vec_g)
        x0 = self.model.set_x0(x0)
        if tau is not None and isinstance(tau, float):
            self.llh.tau = tau
        n_steps = self.n_used_steps + self.n_burn_steps
        pos_x0 = np.zeros((self.n_walker, self.n_dims_f), dtype=float)
        for i, x0_i in enumerate(x0):
            pos_x0[:, i] = self.random_state.poisson(x0_i, size=self.n_walker)
        sampler = self.__initiallize_mcmc__()
        vec_f = self.__run_mcmc__(sampler, pos_x0, n_steps)
        return vec_f

    def __initiallize_mcmc__(self):
        return emcee.EnsembleSampler(nwalkers=self.n_walker,
                                     dim=self.n_dims_f,
                                     lnpostfn=self.llh.evaluate_llh)

    def __run_mcmc__(self, sampler, x0, n_steps):
        sampler.run_mcmc(pos0=x0,
                         N=n_steps,
                         rstate0=self.random_state)
        samples = sampler.chain[:, self.n_burn_steps:, :]
        samples = samples.reshape((-1, self.n_dims_f))

        probs = sampler.lnprobability[:, self.n_burn_steps:]
        probs = probs.reshape((-1))
        idx_max = np.argmax(probs)
        return samples[idx_max, :]


class LLHSolutionHybrid(LLHSolutionMCMC, LLHSolutionMinimizer):
    name = 'LLHSolutionHybrid'

    def __init__(self, n_walker=100, n_used_steps=2000, n_burn_steps=1000):
        super(LLHSolutionHybrid, self).__init__(n_walker=n_walker,
                                                n_used_steps=n_used_steps,
                                                n_burn_steps=n_burn_steps)
        self.bounds = None

    def initialize(self, vec_g, model, bounds=None):
        super(LLHSolutionHybrid, self).initialize(vec_g=vec_g,
                                                  model=model)
        if bounds is True:
           self.bounds = model.generate_bounds(vec_g)
        else:
            self.bounds = None

    def run(self,
            vec_g,
            model,
            tau,
            bounds=None,
            x0=None,
            n_walker=100,
            initial_steps=500,
            additional_steps=50):
        pass

class LLHSolutionDifferentialEvolution(Solution):
    name = 'LLHSolutionMinimizer'

    def run(self, vec_g, model, tau, x0=None, bounds=None):
        self.initialize()
        super(LLHSolutionDifferentialEvolution, self).run()
        if bounds is True:
            bounds = model.generate_bounds(vec_g)
        if x0 is None:
            x0 = model.generate_x0(vec_g)
        x0 = model.set_x0(x0)
        LLH = LLHThikonov(g=vec_g, linear_model=model, tau=tau)
        solution = differential_evolution(func=LLH.evaluate_llh,
                                          bounds=bounds)
        hess_matrix = LLH.evaluate_hesse_matrix(solution.x)
        V_f_est = linalg.inv(hess_matrix)
        return solution.x, V_f_est
