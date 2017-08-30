import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import minimize

import emcee

from ..model import LinearModel
from .error_calculation import calc_feldman_cousins_errors_binned


class Solution(object):
    name = 'Solution'
    status_need_for_fit = 0

    def __init__(self, random_state=None):
        self.status = -1

    def initialize(self):
        self.status = 0

    def set_x0_and_bounds(self):
        if self.status_need_for_fit == 0:
            warnings.warn("{} doesn't use x0 and bounds!")
        else:
            self.status = 1

    def fit(self):
        if self.status < 0 and self.status_need_for_fit == 0:
            raise RuntimeError("Solution has to be intilized. "
                               "Run 'Solution.initialize' first!")
        if self.status < 1 and self.status_need_for_fit == 1:
            raise RuntimeError("Solution has to be intilized and a x0 plus "
                               " have to be provided.Run 'Solution.initialize'"
                               " and 'Solution.set_x0_and_bounds' first!")


class SVDSolution(Solution):
    name = 'SVDSolution'
    status_need_for_fit = 0

    def __init__(self):
        super(SVDSolution, self).__init__()
        self.original_singular_values = None
        self.used_singular_values = None
        self.tau = None
        self.vec_g = None
        self.model = None

    def initialize(self, model, vec_g, tau=None):
        super(SVDSolution, self).initialize()
        if not isinstance(model, LinearModel):
            raise ValueError("'model' has to be of type LinearModel!")
        self.vec_g = vec_g
        self.model = model
        if tau is None:
            self.tau = np.ones(model.dim_f)
        elif isinstance(tau, int):
            if tau > model.dim_f:
                warnings.warn('Number of used singular values is '
                                 'greater than the total number of '
                                 'singular values. The solution will be '
                                 'unregularized!')
                self.tau = np.ones(model.dim_f)
            else:
                self.tau = np.ones(model.dim_f)
                self.tau[tau:] = 0
        elif callable(tau):
            self.tau = tau(np.arange(model.dim_f))
        else:
            raise ValueError("'tau' as to be either None, int or callable!")

    def fit(self):
        super(SVDSolution, self).fit()
        A = self.model.A
        U, S_values, V = linalg.svd(A)
        order = np.argsort(S_values)[::-1]
        S_inv = np.zeros((self.model.dim_f, self.model.dim_g))
        for i, idx in enumerate(order):
            S_inv[i, i] = 1. / np.real(S_values[idx]) * self.tau[i]
        A_inv = np.dot(V.T, np.dot(S_inv, U.T))
        vec_f = np.dot(self.vec_g, A_inv.T)
        vec_f = np.real(vec_f)
        V_y = np.diag(self.vec_g)
        V_f_est = np.real(np.dot(A_inv, np.dot(V_y, A_inv.T)))
        factor = np.sum(self.vec_g) / np.sum(vec_f)
        vec_f *= factor
        V_f_est *= factor
        return vec_f, V_f_est


class LLHSolutionMinimizer(Solution):
    name = 'LLHSolutionMinimizer'
    status_need_for_fit = 1

    def __init__(self):
        super(LLHSolutionMinimizer, self).__init__()
        self.llh = None
        self.vec_g = None
        self.bounds = None
        self.model = None

    def initialize(self, model, llh):
        super(LLHSolutionMinimizer, self).initialize()
        self.llh = llh
        self.vec_g = llh.vec_g
        self.model = model

    def set_x0_and_bounds(self, x0=None, bounds=False):
        super(LLHSolutionMinimizer, self).set_x0_and_bounds()
        if x0 is None:
            x0 = self.model.generate_fit_x0(self.vec_g)
        if bounds is None:
            bounds = self.model.generate_fit_bounds(self.vec_g)
        elif isinstance(bounds, bool):
            if bounds:
                bounds = self.model.generate_fit_bounds(self.vec_g)
            else:
                bounds = None
        self.x0 = x0
        self.bounds = bounds

    def fit(self, constrain_N=True):
        super(LLHSolutionMinimizer, self).fit()
        if constrain_N:
            cons = (
                {'type': 'eq',
                 'fun': lambda x: np.absolute(np.sum(x) - np.sum(self.x0))})
        else:
            cons = ()
        solution = minimize(fun=self.llh.evaluate_llh,
                            x0=self.x0,
                            bounds=self.bounds,
                            method='SLSQP',
                            constraints=cons)
        try:
            hess_matrix = self.llh.evaluate_hesse_matrix(solution.x)
            V_f_est = linalg.inv(hess_matrix)
        except NotImplementedError:
            V_f_est = None
        return solution, V_f_est


class LLHSolutionMCMC(Solution):
    name = 'LLHSolutionMCMC'
    status_need_for_fit = 1

    def __init__(self,
                 n_walker=100,
                 n_used_steps=2000,
                 n_burn_steps=1000,
                 n_threads=1,
                 random_state=None):
        super(LLHSolutionMCMC, self).__init__()
        if not isinstance(random_state, np.random.mtrand.RandomState):
            random_state = np.random.mtrand.RandomState(random_state)
        self.random_state = random_state

        self.n_walker = n_walker
        self.n_used_steps = n_used_steps
        self.n_burn_steps = n_burn_steps
        self.n_threads = n_threads

        self.x0 = None

    def initialize(self, model, llh):
        super(LLHSolutionMCMC, self).initialize()
        self.llh = llh
        self.vec_g = llh.vec_g
        self.model = model


    def set_x0_and_bounds(self, x0=None, bounds=False):
        super(LLHSolutionMCMC, self).set_x0_and_bounds()
        if x0 is None:
            x0 = self.model.generate_fit_x0(self.vec_g)
        self.x0 = x0
        if bounds is not None and bounds:
            warnings.warn("'bounds' have no effect or MCMC!")

    def fit(self):
        super(LLHSolutionMCMC, self).fit()
        n_steps = self.n_used_steps + self.n_burn_steps
        pos_x0 = np.zeros((self.n_walker, self.model.dim_f), dtype=float)
        for i, x0_i in enumerate(self.x0):
            pos_x0[:, i] = self.random_state.poisson(x0_i, size=self.n_walker)
        sampler = self.__initiallize_mcmc__()
        vec_f, samples, probs = self.__run_mcmc__(sampler, pos_x0, n_steps)
        sigma_vec_f = calc_feldman_cousins_errors_binned(vec_f, samples)
        return vec_f, sigma_vec_f, samples, probs

    def __initiallize_mcmc__(self):
        return emcee.EnsembleSampler(nwalkers=self.n_walker,
                                     dim=self.model.dim_f,
                                     lnpostfn=self.llh.evaluate_llh,
                                     threads=self.n_threads)

    def __run_mcmc__(self, sampler, x0, n_steps):
        sampler.run_mcmc(pos0=x0,
                         N=n_steps,
                         rstate0=self.random_state.get_state())
        samples = sampler.chain[:, self.n_burn_steps:, :]
        samples = samples.reshape((-1, self.model.dim_f))

        probs = sampler.lnprobability[:, self.n_burn_steps:]
        probs = probs.reshape((-1))
        idx_max = np.argmax(probs)
        if hasattr(self.model, 'transform_vec_fit'):
            samples = self.model.transform_vec_fit(samples)
        return samples[idx_max, :], samples, probs
