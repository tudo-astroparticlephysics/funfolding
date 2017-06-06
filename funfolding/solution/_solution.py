import logging

import numpy as np
from scipy import linalg
from scipy.optimize import minimize, differential_evolution

import emcee

from ..model import Model, LinearModel


class Solution(object):
    name = 'Solution'
    status_need_for_fit = 0

    def __init__(self, random_state=None):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initilized {}'.format(self.name))
        self.status = -1

    def initialize(self):
        self.logger.debug('Initilizing the Solution!')
        self.status = 0

    def set_x0_and_bounds(self):
        self.logger.debug('Initilizing x0 and bounds!')
        if self.status_need_for_fit == 0:
            self.logger.warn("{} doesn't use x0 and bounds!")
        else:
            self.status = 1

    def fit(self):
        self.logger.debug('Running Solution!')
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

    def initialize(self, llh, model, vec_g, tau=None):
        super(SVDSolution, self).initialize()
        if not isinstance(model, LinearModel):
            raise ValueError("'model' has to be of type LinearModel!")
        if tau is None:
            self.tau = np.ones(model.dim_f)
        elif isinstance(tau, int):
            if tau >= model.dim_f:
                self.logger.warn('Number of used singular values is '
                                 'greater equal to the total number of '
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
        S_inv = np.diag(1 / S_values[order] * self.tau)
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

    def __init__(self):
        super(LLHSolutionMinimizer, self).__init__()
        self.llh = None
        self.vec_g = None
        self.bounds = None
        self.model = None

    def initialize(self, vec_g, model, bounds=True):
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
        return solution, V_f_est

    def __run_minimization__(self, x0):
        cons = ({'type': 'eq', 'fun': lambda x: np.absolute(np.sum(x) -
                                                            np.sum(x0))})
        solution = minimize(fun=self.llh.evaluate_llh,
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
                 random_state=None,
                 n_threads=1):
        super(LLHSolutionMCMC, self).__init__(random_state=random_state)
        self.n_walker = n_walker
        self.n_used_steps = n_used_steps
        self.n_burn_steps = n_burn_steps
        self.llh = None
        self.vec_g = None
        self.model = None
        self.n_dims_f = None
        self.n_threads = 1

    def initialize(self, vec_g, model, bounds=True):
        super(LLHSolutionMCMC, self).initialize()
        self.llh = LLHThikonov(g=vec_g,
                               linear_model=model,
                               neg_llh=False,
                               N_prior=False)
        self.n_dims_f = model.dim_f
        self.vec_g = vec_g
        self.model = model
        if bounds is True:
            self.bounds = model.generate_bounds(vec_g)
        else:
            self.bounds = None

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
        vec_f, samples, probs = self.__run_mcmc__(sampler, pos_x0, n_steps)

        return vec_f, samples, probs

    def __initiallize_mcmc__(self):
        return emcee.EnsembleSampler(nwalkers=self.n_walker,
                                     dim=self.n_dims_f,
                                     lnpostfn=self.llh.evaluate_llh,
                                     threads=self.n_threads)

    def __run_mcmc__(self, sampler, x0, n_steps):
        sampler.run_mcmc(pos0=x0,
                         N=n_steps,
                         rstate0=self.random_state)
        print(sampler.chain.shape)
        samples = sampler.chain[:, self.n_burn_steps:, :]
        samples = samples.reshape((-1, self.n_dims_f))

        probs = sampler.lnprobability[:, self.n_burn_steps:]
        probs = probs.reshape((-1))
        idx_max = np.argmax(probs)
        samples = self.model.transform(samples)
        return samples[idx_max, :], samples, probs


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
