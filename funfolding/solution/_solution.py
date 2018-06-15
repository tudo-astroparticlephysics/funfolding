import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import six

import emcee

from ..model import LinearModel
from . import error_calculation as ec
from .likelihood import StandardLLH, StepLLH


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
        if isinstance(self.llh, StandardLLH):
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
        elif isinstance(self.llh, StepLLH):
            if x0 is None:
                x0 = 1.
            if bounds is None:
                bounds = None
            elif isinstance(bounds, bool):
                if bounds:
                    bounds = [[0., np.inf]]
                else:
                    bounds = None
            self.x0 = x0
            self.bounds = bounds

    def fit(self, constrain_N=True, calc_inv_hessian=True):
        super(LLHSolutionMinimizer, self).fit()
        if constrain_N:
            cons = ({
                'type': 'eq',
                'fun': lambda x: np.absolute(np.sum(x) - np.sum(self.x0))})
        else:
            cons = ()
        solution = minimize(fun=self.llh.evaluate_neg_llh,
                            x0=self.x0,
                            bounds=self.bounds,
                            method='SLSQP',
                            constraints=cons,
                            options={'maxiter': 500})
        if isinstance(self.llh, StandardLLH):
            V_f_est = None
            if calc_inv_hessian:
                try:
                    hess_matrix = self.llh.evaluate_neg_hessian(solution.x)
                    V_f_est = linalg.inv(hess_matrix)
                except NotImplementedError:
                    pass
                except ValueError:
                    warnings.warn('Inversion of the Hessian matrix failed!')
                    pass
            return solution, V_f_est
        elif isinstance(self.llh, StepLLH):
            vec_f_est = self.llh.generate_vec_f_est(solution.x)
            return solution, vec_f_est


class LLHSolutionGradientDescent(LLHSolutionMinimizer):
    name = 'LLHSolutionGradientDescent'
    status_need_for_fit = 1

    def __init__(self, n_steps=500, gamma=0.01):
        super(LLHSolutionGradientDescent, self).__init__()
        self.n_steps = n_steps
        if gamma <= 0.:
            raise ValueError('\'gamma\' has to be > 0!')
        self.gamma = gamma

    def fit(self):
        super(LLHSolutionMinimizer, self).fit()

        x = np.zeros((self.n_steps, len(self.x0)))
        llh = np.zeros(self.n_steps)
        gradient = np.zeros((self.n_steps, len(self.x0)))
        hessian = np.zeros((self.n_steps, len(self.x0), len(self.x0)))

        x[0, :] = self.x0
        llh[0] = self.llh.evaluate_llh(self.x0)
        gradient[0, :] = self.llh.evaluate_gradient(self.x0)
        hessian[0, :, :] = self.llh.evaluate_hessian(self.x0)

        for i in range(1, self.n_steps):
            H_inv = linalg.inv(hessian[i - 1])
            delta_x = -np.dot(H_inv, gradient[i - 1, :]) * self.gamma
            x[i, :] = x[i - 1, :] + delta_x
            x[i, x[i, :] < 0] = 0.
            llh[i] = self.llh.evaluate_llh(x[i, :])
            gradient[i, :] = self.llh.evaluate_gradient(x[i, :])
            hessian[i, :, :] = self.llh.evaluate_hessian(x[i, :])
        return x, llh, gradient, hessian


class LLHSolutionMCMC(Solution):
    name = 'LLHSolutionMCMC'
    status_need_for_fit = 1
    available_errors_calcs = (
        'bayesian',
        'feldmann_unbinned',
        'feldmann_binned',
        'llh_min_max')

    def __init__(self,
                 error_calc='bayesian',
                 n_walkers=100,
                 n_used_steps=2000,
                 n_burn_steps=1000,
                 random_state=None):
        super(LLHSolutionMCMC, self).__init__()
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state
        if error_calc.lower() not in self.available_errors_calcs:
            raise ValueError(
                '{} invalid setting for error calculatioin'.format(error_calc))
        self.error_calc = error_calc.lower()

        self.n_walkers = n_walkers
        self.n_used_steps = n_used_steps
        self.n_burn_steps = n_burn_steps

        self.x0 = None

    def initialize(self, model, llh):
        super(LLHSolutionMCMC, self).initialize()
        self.llh = llh
        self.vec_g = llh.vec_g
        self.model = model

    def set_x0_and_bounds(self, x0=None, bounds=False, min_x0=0.5):
        super(LLHSolutionMCMC, self).set_x0_and_bounds()
        if x0 is None:
            x0 = self.model.generate_fit_x0(self.vec_g, size=None)
        self.x0 = x0
        self.min_x0 = min_x0
        if bounds is not None and bounds:
            warnings.warn("'bounds' have no effect or MCMC!")

    def fit(self,
            thin=None,
            error_interval_sigma=1.,
            error_interval_sigma_limits=1.64):
        super(LLHSolutionMCMC, self).fit()
        n_steps = self.n_used_steps + self.n_burn_steps

        pos_x0 = np.zeros(
            (self.n_walkers, self.model.dim_fit_vector),
            dtype=float
        )

        llh_pos_x0 = np.zeros(self.n_walkers, dtype=float)
        llh_pos_x0[:] = np.inf * -1.
        while any(np.isinf(llh_pos_x0)):
            idx = np.isinf(llh_pos_x0)
            n_new = np.sum(idx)
            new_pos_x0 = self.model.generate_fit_x0(
                vec_g=self.llh.vec_g,
                vec_f_0=self.x0,
                size=n_new)
            pos_x0[idx, :] = new_pos_x0
            llh_pos_x0[idx] = np.array([self.llh(new_pos_x0[i, :])
                                        for i in range(n_new)])

        for i in range(self.n_walkers):
            s = ''
            pos_x0_i = pos_x0[i, :]
            for x_i in pos_x0_i:
                s += '\t{:.2f}'.format(x_i)
        sampler = self.__initiallize_mcmc__()
        sampler.run_mcmc(initial_state=pos_x0,
                         nsteps=n_steps,
                         rstate0=self.random_state.get_state())
        try:
            autocorr_time = sampler.get_autocorr_time()
            trustworthy = True
        except emcee.autocorr.AutocorrError as e:
            autocorr_time = e.tau
            trustworthy = False
        autocorr_time = (autocorr_time, trustworthy)
        sample_flat = sampler.get_chain(flat=True,
                                        discard=self.n_burn_steps)
        probs_flat = sampler.get_log_prob(flat=True,
                                          discard=self.n_burn_steps)
        if self.error_calc == 'bayesian':
            vec_fit_params, sigma_vec_f = ec.bayesian_parameter_estimation(
                sample=sample_flat,
                sigma=error_interval_sigma,
                sigma_limits=error_interval_sigma_limits,
                n_nuissance=self.model.n_nuissance_parameters)
        else:
            idx_max = np.argmax(probs_flat)
            vec_fit_params = sample_flat[idx_max, :]
            if self.error_calc == 'feldmann_unbinned':
                sigma_vec_f = ec.calc_feldman_cousins_errors(
                    best_fit=vec_fit_params,
                    sample=sample_flat,
                    sigma=error_interval_sigma,
                    sigma_limits=error_interval_sigma_limits,
                    n_nuissance=self.model.n_nuissance_parameters)
            if self.error_calc == 'feldmann_binned':
                sigma_vec_f = ec.calc_feldman_cousins_errors_binned(
                    best_fit=vec_fit_params,
                    sample=sample_flat,
                    sigma=error_interval_sigma,
                    sigma_limits=error_interval_sigma_limits,
                    n_nuissance=self.model.n_nuissance_parameters)
            elif self.error_calc == 'llh_min_max':
                sigma_vec_f = ec.calc_errors_llh(
                    sample=sample_flat,
                    probs=probs_flat,
                    sigma=error_interval_sigma,
                    sigma_limits=error_interval_sigma_limits,
                    n_nuissance=self.model.n_nuissance_parameters)
        if thin is not None:
            if isinstance(thin, six.string_types):
                if thin.lower() == 'autocorr':
                    thin = int(np.max(autocorr_time[0]) + 0.5)
            if isinstance(thin, int):
                sample = sampler.get_chain(thin=thin,
                                           flat=True,
                                           discard=self.n_burn_steps)
                probs = sampler.get_log_prob(thin=thin,
                                             flat=True,
                                             discard=self.n_burn_steps)
            else:
                raise ValueError("'thin' has to be either an int, None or "
                                 "autocorr")
        else:
            sample = sample_flat
            probs = probs_flat

        return vec_fit_params, sigma_vec_f, sample, probs, autocorr_time

    def __initiallize_mcmc__(self):
        return emcee.EnsembleSampler(nwalkers=self.n_walkers,
                                     ndim=self.model.dim_fit_vector,
                                     log_prob_fn=self.llh)

    def __run_mcmc__(self, sampler, x0, n_steps):
        sampler.run_mcmc(initial_state=x0,
                         nsteps=n_steps,
                         rstate0=self.random_state.get_state())

        samples = sampler.get_chain()
        samples = samples[self.n_burn_steps:, :, :]

        probs = sampler.get_log_prob()
        probs = probs[self.n_burn_steps:, :]
        if hasattr(self.model, 'transform_vec_fit'):
            samples = self.model.transform_vec_fit(samples)
        return samples, probs
