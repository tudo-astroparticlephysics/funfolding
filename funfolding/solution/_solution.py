import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import minimize

import emcee
try:
    from pymc.diagnostics import effective_n
    no_pymc = False
except ImportError:
    no_pymc = True

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
            cons = ({
                'type': 'eq',
                'fun': lambda x: np.absolute(np.sum(x) - np.sum(self.x0))})
        else:
            cons = ()
        solution = minimize(fun=self.llh.evaluate_neg_llh,
                            x0=self.x0,
                            bounds=self.bounds,
                            method='SLSQP',
                            constraints=cons)
        try:
            hess_matrix = self.llh.evaluate_neg_hessian(solution.x)
            V_f_est = linalg.inv(hess_matrix)
        except NotImplementedError:
            V_f_est = None
        return solution, V_f_est


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
            H_inv = linalg.inv(hessian[i-1])
            delta_x = -np.dot(H_inv, gradient[i-1, :]) * self.gamma
            x[i, :] = x[i - 1, :] + delta_x
            llh[i] = self.llh.evaluate_llh(x[i, :])
            gradient[i, :] = self.llh.evaluate_gradient(x[i, :])
            hessian[i, :, :] = self.llh.evaluate_hessian(x[i, :])
        return x, llh, gradient, hessian


class LLHSolutionMCMC(Solution):
    name = 'LLHSolutionMCMC'
    status_need_for_fit = 1

    def __init__(self,
                 n_walkers=100,
                 n_used_steps=2000,
                 n_burn_steps=1000,
                 n_threads=1,
                 random_state=None):
        super(LLHSolutionMCMC, self).__init__()
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.n_walkers = n_walkers
        self.n_used_steps = n_used_steps
        self.n_burn_steps = n_burn_steps
        self.n_threads = n_threads

        self.x0 = None

    def initialize(self, model, llh):
        super(LLHSolutionMCMC, self).initialize()
        self.llh = llh
        self.vec_g = llh.vec_g
        self.model = model

    def set_x0_and_bounds(self, x0=None, bounds=False, min_x0=0.5):
        super(LLHSolutionMCMC, self).set_x0_and_bounds()
        if x0 is None:
            x0 = self.model.generate_fit_x0(self.vec_g)
        self.x0 = x0
        self.min_x0 = min_x0
        if bounds is not None and bounds:
            warnings.warn("'bounds' have no effect or MCMC!")

    def fit(self):
        super(LLHSolutionMCMC, self).fit()
        n_steps = self.n_used_steps + self.n_burn_steps
        pos_x0 = np.zeros((self.n_walkers, self.model.dim_f), dtype=float)
        for i, x0_i in enumerate(self.x0):
            if x0_i < 1.:
                x0_i += self.min_x0
            pos_x0[:, i] = self.random_state.poisson(x0_i, size=self.n_walkers)
        pos_x0[pos_x0 == 0] = self.min_x0
        sampler = self.__initiallize_mcmc__()
        vec_f, samples, probs = self.__run_mcmc__(sampler,
                                                  pos_x0,
                                                  n_steps)
        sigma_vec_f = calc_feldman_cousins_errors_binned(vec_f, samples)
        return vec_f, sigma_vec_f, samples, probs

    def __initiallize_mcmc__(self):
        return emcee.EnsembleSampler(nwalkers=self.n_walkers,
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

    def calc_effective_sample_size(self, sample, n_threads=None):
        '''Function to calculate the effective sample_size.

        Calculation uses effective_n from the pymc.diagonstics module.
        It is based on the
        Internally the sample is reshaped to
        (n_walkers, n_samples_per_walker, dims_f). Changing the
        n_walkers attribute of the instance will break the calculation.

        Parameters
        ----------
        x : array-like, shape=(n_walkers*n_used_steps, dim_f)
            An array containing the reshape samples for all walkers.
            Internally it will be reshaped to m x n x k, where m is
            the number of walkers, n the number of samples, and k the dimension
            of the stochastic.

        n_threads : int or None
            Number of threads used to calculate the effective sample size.
            If None the n_threads from the LLHSolutionMCMC instance is used.

        Returns
        -------
        n_eff : array-like, shape=(dim_f,)
            Return the effective sample size, :math:`\hat{n}_{eff}`

        Notes
        -----
        The diagnostic is computed by:
          .. math:: \hat{n}_{eff} = \frac{mn}}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}
        where :math:`\hat{\rho}_t` is the estimated autocorrelation at lag t,
        and T is the first odd positive integer for which the sum
        :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}`
        is negative.

        References
        ----------
        Gelman et al. (2014)
        '''
        if no_pymc:
            raise ImportError('To call \'calc_effective_sample_size\' '
                              '\'pymc\' has to be installed!')
        if n_threads is None:
            n_threads = self.n_threads
        elif not isinstance(n_threads, int):
            raise ValueError('\'n_threads\' has to be int or None!')
        dim_f = sample.shape[1]
        sample = sample.reshape((self.n_walkers,
                                 self.n_used_steps,
                                 dim_f))

        n_eff = [0] * dim_f
        n_threads = min(dim_f, n_threads)

        if n_threads > 1:
            from concurrent.futures import ProcessPoolExecutor
            import time
            with ProcessPoolExecutor(max_workers=n_threads) as executor:
                def future_callback(future):
                    future_callback.finished += 1
                    if not future.cancelled():
                        i, n_eff_i = future.result()
                        n_eff[i] = n_eff_i
                    else:
                        raise RuntimeError('Subprocess crashed!')
                    future_callback.running -= 1

                future_callback.running = 0
                future_callback.finished = 0
                for i in range(dim_f):
                    while True:
                        if future_callback.running < n_threads:
                            break
                        else:
                            time.sleep(1)
                    future = executor.submit(
                        __effective_n_idx__,
                        x=sample[:, :, i],
                        idx=i)
                    future_callback.running += 1
                    future.add_done_callback(future_callback)
            return n_eff
        else:
            return effective_n(sample)


def __effective_n_idx__(idx, x):
    return idx, effective_n(x)
