import numpy as np
import six

from ..model import LinearModel, Model


def create_C_thikonov(n_dims,
                      ignore_n_bins_low=0,
                      ignore_n_bins_high=0):
    C = np.zeros((n_dims, n_dims))
    C[ignore_n_bins_low, ignore_n_bins_low] = -1
    C[ignore_n_bins_low, ignore_n_bins_low + 1] = 1
    idx_N = n_dims - 1 - ignore_n_bins_high
    C[idx_N, idx_N] = -1
    C[idx_N, idx_N - 1] = 1
    for i in range(1 + ignore_n_bins_low, idx_N):
        C[i, i] = -2.
        C[i, i - 1] = 1
        C[i, i + 1] = 1
    return C


class LLH(object):
    name = 'LLH'
    status_needed_for_eval = 0

    def __init__(self):
        self.status = -1

        self.vec_g = None
        self.model = None

        self.gradient_defined = False
        self.hessian_matrix_defined = False

    def initialize(self):
        self.status = 0

    def evaluate_llh(self):
        if self.status < self.status_needed_for_eval:
            msg = 'LLH needs status {} not {} to be evaluated'.format(
                self.status_needed_for_eval, self.status)
            raise RuntimeError(msg)

    def evaluate_neg_llh(self, f):
        return self.evaluate_llh(f) * -1.

    def evaluate_gradient(self):
        if self.gradient_defined:
            if self.status < self.status_needed_for_eval:
                msg = 'LLH needs status {} not {} to be evaluated'.format(
                    self.status_needed_for_eval, self.status)
                raise RuntimeError(msg)
        else:
            raise NotImplementedError("Gradients are not implemented!")

    def evaluate_neg_gradient(self, f):
        return self.evaluate_gradient(f) * -1.

    def evaluate_hessian(self):
        if self.hessian_matrix_defined:
            if self.status < self.status_needed_for_eval:
                msg = 'LLH needs status {} not {} to be evaluated'.format(
                    self.status_needed_for_eval, self.status)
                raise RuntimeError(msg)
        else:
            raise NotImplementedError("hessian Matrix is not implemented!")

    def evaluate_neg_hessian(self, f):
        return self.evaluate_hessian(f) * -1.

    def __call__(self, f):
        return self.evaluate_llh(f)


class StandardLLH(LLH):
    name = 'StandardLLH'
    status_need_for_eval = 0

    def __init__(self,
                 tau=None,
                 C='thikonov',
                 reg_factor_f=None,
                 log_f=False,
                 log_f_offset=1):
        super(StandardLLH, self).__init__()
        self.C = C
        self.tau = tau
        self.log_f_reg = log_f
        self.reg_factor_f = reg_factor_f
        self.log_f_offset = log_f_offset

    def initialize(self,
                   vec_g,
                   model,
                   ignore_n_bins_low=0,
                   ignore_n_bins_high=0):
        super(StandardLLH, self).initialize()
        if not isinstance(model, Model):
            raise ValueError("'model' has to be of type Model!")
        self.model = model
        self.vec_g = vec_g

        eff_f_length = model.dim_f - ignore_n_bins_low - ignore_n_bins_high

        if self.reg_factor_f is None:
            self.reg_factor_f = np.ones(eff_f_length)

        if self.tau is None:
            self._tau = None
        elif isinstance(self.tau, six.string_types):
            if self._tau.lower() == 'None':
                self._tau = None
        else:
            self._f_slice = slice(ignore_n_bins_low,
                                  model.dim_f - ignore_n_bins_high)
            if len(self.reg_factor_f) != eff_f_length:
                raise ValueError(
                    'Length of f used for regularization != length of '
                    'reg_factor_f (f: {}, reg_factor_f: {})'.format(
                        eff_f_length, len(self.reg_factor_f)))
            if isinstance(self.tau, float):
                if self.tau <= 0.:
                    self._tau = None
                else:
                    self._tau = np.ones(eff_f_length) * self.tau
            elif callable(self.tau):
                self._tau = self.tau(np.arange(model.dim_f))
                self._tau = self._tau[self._f_slice]
            elif isinstance(self.tau, np.array) or \
                    isinstance(self.tau, list) or \
                    isinstance(self.tau, tuple):
                if len(self.tau) == model.dim_f:
                    self._tau = np.array(self.tau)
                else:
                    raise ValueError(
                        "Length of 'tau'={} invalid! {} needed".format(
                            len(self.tau),
                            model.dim_f))
            else:
                raise ValueError("'tau' as to be either None, float, array or "
                                 "callable!")
            if self._tau is not None:
                m_C = None
                if isinstance(self.C, six.string_types):
                    if self.C.lower() == 'thikonov' or self.C.lower() == '2':
                        m_C = create_C_thikonov(
                            eff_f_length)
                elif isinstance(self.C, int):
                    if self.C == 2:
                        m_C = create_C_thikonov(eff_f_length)
                if m_C is None:
                    raise ValueError("{} invalid option for 'C'".format(
                        self.C))
                self._C = np.dot(np.dot(m_C, np.diag(1 / self._tau)), m_C)

        if isinstance(model, LinearModel):
            self.gradient_defined = True
            self.hessian_matrix_defined = True

    def evaluate_llh(self, fit_params):
        super(StandardLLH, self).evaluate_llh()
        g_est, f, f_reg = self.model.evaluate(fit_params)
        if any(g_est < 0) or any(f < 0):
            return np.inf * -1
        poisson_part = np.sum(self.vec_g * np.log(g_est) - g_est)
        if self._tau is not None:
            f_reg = f_reg[self._f_slice]
            if self.log_f_reg:
                f_reg_used = np.log10((f_reg + self.log_f_offset) *
                                      self.reg_factor_f)
            else:
                f_reg_used = f_reg * self.reg_factor_f
            reg_part = 0.5 * np.dot(
                np.dot(f_reg_used.T, self._C), f_reg_used)
        else:
            reg_part = 0
        return poisson_part - reg_part

    def create_pymc_model(self, x0=None):
        super(StandardLLH, self).evaluate_llh()
        import pymc3 as pm
        import theano
        model = pm.Model()

        if x0 is None:
            x0 = np.sum(self.vec_g) / self.model.dim_f
            x0 = np.ones(self.model.dim_f, dtype=float) * x0

        with model:
            A = theano.shared(self.model.A)
            vec_g = theano.shared(self.vec_g)
            f = pm.Uniform('f',
                           testval=x0,
                           lower=0,
                           upper=np.sum(self.vec_g),
                           shape=self.model.dim_f)
            g_est = theano.tensor.dot(A, f)
            poisson_part = pm.Poisson('poisson_part_llh',
                                      mu=g_est,
                                      observed=vec_g)
            if self.tau > 0:
                reg_factor_f = theano.shared(self.reg_factor_f)
                _C = theano.shared(self._C)
                if self.log_f_reg:
                    def calc_reg_part(f_reg):
                        f_reg_used = theano.tensor.log10(
                            (f_reg + 1) * reg_factor_f)
                        return 0.5 * theano.tensor.dot(
                            theano.tensor.dot(f_reg_used.T, _C),
                            f_reg_used)
                else:
                    def calc_reg_part(f_reg):
                        return 0.5 * theano.tensor.dot(
                            theano.tensor.dot(f_reg.T, _C),
                            f_reg)
                reg_part = pm.Deterministic('reg_part', calc_reg_part(f))
            else:
                reg_part = 0.
            pm.Deterministic('logp',
                             theano.tensor.sum(poisson_part) - reg_part)
        return model

    def evaluate_gradient(self, f):
        super(StandardLLH, self).evaluate_gradient()
        g_est, f, f_reg = self.model.evaluate(f)
        part_b = np.sum(self.model.A, axis=0)
        h_unreg = np.sum(self.model.A.T * self.vec_g * (1 / g_est), axis=1)
        h_unreg -= part_b
        if self._tau is not None:
            if self.log_f_reg:
                reg_part = np.zeros(self.model.dim_f)
                denom_f = f_reg + self.log_f_offset
                nom_f = np.log(denom_f * self.reg_factor_f)
                ln_10_squared = np.log(10)**2
                pre = np.zeros((self.model.dim_f,
                                self.model.dim_f))
                for i in range(self.model.dim_f):
                    for j in range(self.model.dim_f):
                        pre[i, j] = self._C[i, j] * nom_f[i]
                        pre[i, j] /= ln_10_squared * denom_f[i]
                for i in range(self.model.dim_f):
                    reg_part[i] = np.sum(pre[i, :])
                    reg_part[i] += np.sum(pre[:, i])
            else:
                reg_part = np.dot(self._C, f_reg * self.reg_factor_f)
        else:
            reg_part = 0.
        return h_unreg - reg_part

    def evaluate_hessian(self, f):
        super(StandardLLH, self).evaluate_hessian()
        g_est, f, f_reg = self.model.evaluate(f)
        H_unreg = -np.dot(np.dot(self.model.A.T,
                                 np.diag(self.vec_g / g_est**2)),
                          self.model.A)
        if self._tau is not None:
            if self.log_f_reg:
                reg_part = self._C + self._C.T
                f_reg = f_reg + self.log_f_offset
                ln_f_reg_used = np.log(f_reg * self.reg_factor_f)
                pre_diag = np.sum(np.dot(reg_part, np.diag(ln_f_reg_used)),
                                  axis=1)
                reg_part -= np.diag(pre_diag)
                denom = np.outer(f_reg, f_reg) * np.log(10)**2
                reg_part /= denom
            else:
                reg_part = self._C
            reg_part /= denom
        else:
            reg_part = 0.

        return H_unreg - reg_part


class StepLLH(LLH):
    name = 'StepLLH'
    status_needed_for_eval = 1

    def __init__(self):
        super(StepLLH, self).__init__()
        self.__previous_f = None
        self.__step = None

    def set_fs(self, previous_f, current_f):
        self.status = 1
        self.__previous_f = previous_f
        self.__step = current_f - previous_f

    def generate_vec_f_est(self, a):
        return self.__previous_f + a * self.__step

    def initialize(self,
                   vec_g,
                   model):
        super(StepLLH, self).initialize()
        if not isinstance(model, Model):
            raise ValueError("'model' has to be of type Model!")
        self.model = model
        self.vec_g = vec_g
        self.N = np.sum(vec_g)

        if isinstance(model, LinearModel):
            self.gradient_defined = True
            self.hessian_matrix_defined = True

    def evaluate_llh(self, a):
        super(StepLLH, self).evaluate_llh()
        f = self.__previous_f + a * self.__step
        g_est, f, _ = self.model.evaluate(f)
        if any(g_est < 0) or any(f < 0):
            return np.inf * -1
        poisson_part = np.sum(self.vec_g * np.log(g_est) - g_est)
        return poisson_part

    def evaluate_gradient(self, a):
        super(StepLLH, self).evaluate_gradient()
        f = self.__previous_f + a * self.__step
        g_est, f, _ = self.model.evaluate(f)
        A_delta = np.dot(self.model.A, self.__step)
        part_b = np.sum(A_delta, axis=0)
        h_unreg = np.sum(A_delta * (1 / g_est), axis=1)
        h_unreg -= part_b
        return h_unreg

    def evaluate_hessian(self, a):
        super(StepLLH, self).evaluate_hessian()
        f = self.__previous_f + a * self.__step
        g_est, f, _ = self.model.evaluate(f)
        A_delta = np.dot(self.model.A, self.__step)
        H_unreg = - (self.vec_g * A_delta**2) / g_est**2
        return H_unreg


class LLHThikonovForLoops(LLH):
    name = 'StepLLH'
    status_needed_for_eval = 1

    def __init__(self, g, linear_model, tau):
        import warnings
        warnings.warn('{} is old and untested!'.format(self.name))
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

    def evaluate_hessian(self, f):
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


class SystematicLLH(StandardLLH):
    name = 'SystematicLLH'
    status_needed_for_eval = 0

    def __init__(self,
                 tau=None,
                 C='thikonov',
                 reg_factor_f=None,
                 log_f=False,
                 log_f_offset=1):
        super(SystematicLLH, self).__init__(
            tau=tau,
            C=C,
            reg_factor_f=reg_factor_f,
            log_f=log_f)
        self.log_f_offset = log_f_offset

    def initialize(self,
                   vec_g,
                   model,
                   ignore_n_bins_low=0,
                   ignore_n_bins_high=0):
        super(SystematicLLH, self).initialize(
            vec_g=vec_g,
            model=model,
            ignore_n_bins_high=ignore_n_bins_high,
            ignore_n_bins_low=ignore_n_bins_low)

        self.gradient_defined = False
        self.hessian_matrix_defined = False

    def evaluate_llh(self, fit_params):
        super(StandardLLH, self).evaluate_llh()
        g_est, f, f_reg = self.model.evaluate(fit_params)
        try:
            if any(g_est < 0) or any(f < 0):
                return np.inf * -1
        except TypeError:
            print(g_est, f, f_reg)
            raise TypeError
        poisson_part = np.sum(self.vec_g * np.log(g_est) - g_est)
        if self._tau is not None:
            f_reg = f_reg[self._f_slice]
            if self.log_f_reg:
                f_reg_used = np.log10((f_reg + self.log_f_offset) *
                                      self.reg_factor_f)
            else:
                f_reg_used = f_reg * self.reg_factor_f
                if any(np.isnan(f_reg_used)):
                    raise ValueError('f_reg_used got a nan!')
            reg_part = 0.5 * np.dot(
                np.dot(f_reg_used.T, self._C), f_reg_used)
        else:
            reg_part = 0
        p = poisson_part - reg_part
        fit_params_pointer = 0
        for (_, lnprob_prior, n_parameters) in self.model.x0_distributions:
            if lnprob_prior is not None:
                fit_params_slice = slice(fit_params_pointer,
                                         fit_params_pointer + n_parameters)
                prior_value = lnprob_prior(fit_params[fit_params_slice])
                p += prior_value
            fit_params_pointer += n_parameters
        return p

    def evaluate_gradient(self, f):
        raise NotImplementedError()

    def evaluate_hessian(self, f):
        raise NotImplementedError()


class StandardLLH_offset_before_log(StandardLLH):
    name = 'StandardLLH_offset_before_log'
    status_needed_for_eval = 0

    def evaluate_llh(self, f):
        super(StandardLLH, self).evaluate_llh()
        g_est, f, f_reg = self.model.evaluate(f)
        if any(g_est < 0) or any(f < 0):
            return np.inf * -1
        poisson_part = np.sum(self.vec_g * np.log(g_est) - g_est)
        if self._tau is not None:
            f_reg_used = f_reg * self.reg_factor_f
            if self.log_f_reg:
                f_reg_used = np.log10(f_reg_used + 1)
            reg_part = 0.5 * np.dot(
                np.dot(f_reg_used.T, self._C), f_reg_used)
        else:
            reg_part = 0
        return poisson_part - reg_part

    def evaluate_gradient(self, f):
        super(StandardLLH, self).evaluate_gradient()
        g_est, f, f_reg = self.model.evaluate(f)
        part_b = np.sum(self.model.A, axis=0)
        h_unreg = np.sum(self.model.A.T * self.vec_g * (1 / g_est), axis=1)
        h_unreg -= part_b
        if self._tau is not None:
            if self.log_f_reg:
                reg_part = np.zeros(self.model.dim_f)
                f_used = f_reg * self.reg_factor_f + 1
                ln_f_used = np.log(f_used)
                ln_10_squared = np.log(10)**2
                pre = np.zeros((self.model.dim_f,
                                self.model.dim_f))
                for i in range(self.model.dim_f):
                    for j in range(self.model.dim_f):
                        pre_part_ij = self.reg_factor_f[i] * self._C[i, j]
                        pre_part_ij *= ln_f_used[j]
                        pre_part_ij /= ln_10_squared * f_used[i]
                        pre[i] += pre_part_ij
                for i in range(self.model.dim_f):
                    reg_part_i = np.sum(pre[i, :])
                    reg_part_i += np.sum(pre[:, i])

            else:
                reg_part = np.dot(self._C, f_reg * self.reg_factor_f)
        else:
            reg_part = 0.
        return h_unreg - reg_part

    def evaluate_hessian(self, f):
        super(StandardLLH, self).evaluate_hessian()
        g_est, f, f_reg = self.model.evaluate(f)
        H_unreg = -np.dot(np.dot(self.model.A.T,
                                 np.diag(self.vec_g / g_est**2)),
                          self.model.A)
        if self._tau is not None:
            if self.log_f_reg:
                reg_part = np.zeros((self.model.dim_f,
                                     self.model.dim_f))
                f_used = f_reg * self.reg_factor_f + 1
                ln_f_used = np.log(f_used)
                ln_10_squared = np.log(10)**2
                for i in range(self.model.dim_f):
                    for j in range(i + 1):
                        r = (self._C[i, j] + self._C[j, i])
                        r *= self.reg_factor_f[i] * self.reg_factor_f[j]
                        r /= ln_10_squared * f_used[i] * f_used[j]
                        if i == j:
                            r_diag = -self.reg_factor_f[j]**2 / ln_10_squared
                            r_diag = f_used**2
                            r_diag = np.sum((self._C[i, :] + self._C[:, i]) *
                                            ln_f_used)
                            reg_part[i, i] = r + r_diag
                        else:
                            reg_part[i, j] = r
                            reg_part[j, i] = r
            else:
                reg_part = self._C
        else:
            reg_part = 0.
        return H_unreg - reg_part
