import warnings
import numpy as np
from scipy import linalg
from scipy import stats


class Model(object):
    name = 'Model'
    status_need_for_eval = 0
    """ Base class for a model. Actual models should inherit from this class.

    In this class the functions that should be implemented by each model are
    defined.

    Attributes
    ----------
    name : str
        Name of the model.

    status : int
        Indicates the status of the model:
            -1 : Instance created. Not filled with values yet.
             0 : Filled with values
             1 : Filled with values and x0 set (optional level).
    """
    def __init__(self):
        self.status = -1
        self.has_background = False

    def initialize(self):
        """This function should be called with all needed values. To actually
        fill all the models with values.
        """
        self.status = 0

    def evaluate(self):
        """Evaluates the model.

        Actual implementation of this functions should return:
            vec_g     : Observable vector
            vec_f     : Solution vector
            vec_f_reg : Vector used in the regularization

        """
        if self.status < 0 and self.status_need_for_eval == 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        if self.status < 1 and self.status_need_for_eval == 1:
            raise RuntimeError("Model has to be intilized and x0 has to be"
                               "set. Run 'model.initialize' and "
                               "'model.set_x0' first!")

    def set_model_x0(self):
        """Some models need to be set up with a x0 for the model. For those .
        models the class parameter 'status_need_for_eval' should be set to 1.
        """
        if self.status < 0:
            raise RuntimeError("Model has to be intilized, before setting x0. "
                               "Run 'model.initialize' first!")
        self.status = 1

    def generate_fit_x0(self):
        """The model should be able to return resonable starting values
        for the fitter.
        """
        if self.status < 0 and self.status_need_for_eval == 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        if self.status < 1 and self.status_need_for_eval == 1:
            raise RuntimeError("Model has to be intilized and x0 has to be"
                               "set. Run 'model.initialize' and "
                               "'model.set_x0' first!")

    def generate_fit_bounds(self):
        """The model should be able to return resonable bounds for the fitter.
        """
        if self.status < 0 and self.status_need_for_eval == 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        if self.status < 1 and self.status_need_for_eval == 1:
            raise RuntimeError("Model has to be intilized and x0 has to be"
                               "set. Run 'model.initialize' and "
                               "'model.set_x0' first!")

    def add_background(self):
        self.has_background = True

    def remove_background(self):
        """Disables the background vector. A stored background vector is
        not deleted.
        """
        self.has_background = False


class LinearModel(Model):
    name = 'LinearModel'
    status_need_for_eval = 0
    """ Basic Linear model:
    g = A * f

    Attributes
    ----------
    name : str
        Name of the model.

    status : int
        Indicates the status of the model:
            -1 : Instance created. Not filled with values yet.
             0 : Filled with values

    dim_g :
        Dimension of the histogrammed observable vector.

    dim_f :
        Dimension of the histogrammed truth vector.

    range_obs : tuple (int, int)
        Tuple containing the lowest and highest bin number used in
        the digitized observable vector. For performance reasons it is
        assumed that all numbers between min and max are used.

    range_truth : tuple (int, int)
        Tuple containing the lowest and highest bin number used in
        the digitized truth vector. For performance reasons it is
        assumed that all numbers between min and max are used.

    A : numpy.array shape=(dim_g, dim_f)
        Response matrix.

    vec_b : numpy.array, shape=(dim_f)
        Observable vector for the background.

    has_background : boolean
        Indicator if self.vec_b should be added to the model evaluationg
    """
    def __init__(self):
        super(LinearModel, self).__init__()
        self.range_obs = None
        self.range_truth = None
        self.A = None
        self.dim_f = None
        self.dim_g = None
        self.vec_b = None
        self.dim_fit_vector = None

    def initialize(self, digitized_obs, digitized_truth, sample_weight=None):
        """

        """
        super(LinearModel, self).initialize()
        self.range_obs = (min(digitized_obs), max(digitized_obs))
        self.range_truth = (min(digitized_truth), max(digitized_truth))
        self.dim_f = self.range_truth[1] - self.range_truth[0] + 1
        self.dim_g = self.range_obs[1] - self.range_obs[0] + 1
        binning_g, binning_f = self.__generate_binning__()
        self.A = np.histogram2d(x=digitized_obs,
                                y=digitized_truth,
                                bins=(binning_g, binning_f),
                                weights=sample_weight)[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = np.dot(self.A, M_norm)
        self.dim_fit_vector = self.dim_f

    def evaluate(self, vec_fit):
        """Evaluating the model for a given vector f

        Parameters
        ----------
        vec_fit : numpy.array, shape=(dim_f,)
            Vector f for which the model should be evaluated.

        Returns
        -------
        vec_g : nump.array, shape=(dim_g,)
            Vector containing the number of events in observable space.
            If background was added the returned vector is A * vec_f + vec_b.

        vec_f : nump.array, shape=(dim_f,)
            Vector used to evaluate A * vec_f

        vec_f_reg : nump.array, shape=(dim_f,)
            Vector that should be passed to the regularization. For the
            BasisLinearModel it is identical to f.
        """
        super(LinearModel, self).evaluate()
        vec_g = np.dot(self.A, vec_fit)
        if self.has_background:
            vec_g += self.vec_b
        return vec_g, vec_fit, vec_fit

    def generate_fit_x0(self, vec_g):
        """Generates a default seed for the minimization.
        The default seed vec_f_0 is a uniform distribution with
        sum(vec_f_0) = sum(vec_g). If background is present the default seed
        is: sum(vec_f_0) = sum(vec_g) - sum(self.vec_b).

        Parameters
        ----------
        vec_g : np.array, shape=(dim_g)
            Observable vector which should be used to get the correct
            normalization for vec_f_0.

        Returns
        -------
        vec_f_0 : np.array, shape=(dim_f)
            Seed vector of a minimization.
        """
        super(LinearModel, self).generate_fit_x0()
        n = self.A.shape[1]
        if self.has_background:
            vec_f_0 = np.ones(n) * (np.sum(vec_g) - np.sum(self.vec_b)) / n
        else:
            vec_f_0 = np.ones(n) * np.sum(vec_g) / n
        return vec_f_0

    def generate_fit_bounds(self, vec_g):
        """Generates a bounds for a minimization.
        The bounds are (0, sum(vec_g)) without background and
        (0, sum(vec_g - self.vec_b)) with background. The bounds are for
        each fit parameter/entry in f.

        Parameters
        ----------
        vec_g : np.array, shape=(dim_g)
            Observable vector which should be used to get the correct
            upper bound

        Returns
        -------
        bounds : list, shape=(dim_f)
            List of tuples with the bounds.
        """
        super(LinearModel, self).generate_fit_bounds()
        n = self.A.shape[1]
        if self.has_background:
            n_events = np.sum(vec_g) - np.sum(self.vec_b)
        else:
            n_events = np.sum(vec_g)
        bounds = [(0, n_events)] * n
        return bounds

    def set_model_x0(self):
        """The LinearModel has no referenz model_x0.
        """
        super(LinearModel, self).set_model_x0()
        warnings.warn('\tx0 has no effect for {}'.format(self.name))

    def evaluate_condition(self, normalize=True):
        """Returns an ordered array of the singular values of matrix A.

        Parameters
        ----------
        normalize : boolean (optional)
            If True the singular values return relativ to the largest
            value.

        Returns
        -------
        S_values : np.array, shape=(dim_f)
            Ordered array of the singular values.
        """
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        U, S_values, V = linalg.svd(self.A)
        if normalize:
            S_values = S_values / S_values[0]
        return S_values

    def __generate_binning__(self):
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        binning_obs = np.linspace(self.range_obs[0],
                                  self.range_obs[1] + 1,
                                  self.dim_g + 1)
        binning_truth = np.linspace(self.range_truth[0],
                                    self.range_truth[1] + 1,
                                    self.dim_f + 1)
        return binning_obs, binning_truth

    def generate_vectors(self,
                         digitized_obs=None,
                         digitized_truth=None,
                         obs_weights=None,
                         truth_weights=None):
        """Returns vec_g, vec_f for digitized values. Either f, g or both
        can be provided to the function.

        Parameters
        ----------
        digitized_obs : np.intarray (optional)
            Array with digitized values form the observable space

        digitized_truth : np.intarray (optinal)
            Array with digitized values for the sought-after quantity.

        Returns
        -------
        vec_g : None or np.array shape=(dim_g)
            None if no digitized_obs was provided otherwise the histrogram
            of digitized_obs.

        vec_f : None or np.array shape=(dim_f)
            None if no digitized_truth was provided otherwise the histrogram
            of digitized_truth.
        """
        binning_obs, binning_truth = self.__generate_binning__()
        if digitized_obs is not None:
            vec_g = np.histogram(digitized_obs,
                                 bins=binning_obs,
                                 weights=obs_weights)[0]
        else:
            vec_g = None
        if digitized_truth is not None:
            vec_f = np.histogram(digitized_truth,
                                 bins=binning_truth,
                                 weights=truth_weights)[0]
        else:
            vec_f = None
        return vec_g, vec_f

    def add_background(self, vec_b):
        """Adds a background vector to the model.

        Parameters
        ----------
        vec_b : numpy.array, shape=(dim_g)
            Vector g which is added to the model evaluation.
        """
        super(LinearModel, self).add_background()
        self.vec_b = vec_b


class BiasedLinearModel(LinearModel):
    name = 'BiasedLinearModel'
    status_need_for_eval = 1
    """Extense the LinearModel with an bias distribtuion model_x0.
    the vec_f is interpreted as element-wise multiple of the model_x0.

    g = A * (model_x0 * vec_fit)

    Internally the model_x0 is normalize in a way that
    vec_fit = [1.] * dim_f
    is transformed to
     vec_f = model_x0 / sum(model_x0) * sum(vec_g).

    Attributes
    ----------
    name : str
        Name of the model.

    model_x0 : np.array, shape=(vec_f)
        Distribtuion which is element-wise multiplied with the vec_fit.

    status : int
        Indicates the status of the model:
            -1 : Instance created. Not filled with values yet.
             0 : Filled with values

    dim_g :
        Dimension of the histogrammed observable vector.

    dim_f :
        Dimension of the histogrammed truth vector.

    range_obs : tuple (int, int)
        Tuple containing the lowest and highest bin number used in
        the digitized observable vector. For performance reasons it is
        assumed that all numbers between min and max are used.

    range_truth : tuple (int, int)
        Tuple containing the lowest and highest bin number used in
        the digitized truth vector. For performance reasons it is
        assumed that all numbers between min and max are used.

    A : numpy.array shape=(dim_g, dim_f)
        Response matrix.

    vec_b : numpy.array, shape=(dim_f)
        Observable vector for the background.

    has_background : boolean
        Indicator if self.vec_b should be added to the model evaluationg
    """
    def __init__(self):
        super(LinearModel, self).__init__()
        self.range_obs = None
        self.range_truth = None
        self.A = None
        self.dim_f = None
        self.dim_g = None
        self.model_x0 = None
        self.model_factor_ = 1.
        self.background_factor_ = 0.
        self.vec_b = None
        self.dim_fit_vector = self.dim_f

    def evaluate(self, vec_fit):
        """Evaluating the model for a given vector f

        Parameters
        ----------
        vec_fit : numpy.array, shape=(dim_f,)
            Vector f for which the model should be evaluated.

        Returns
        -------
        vec_g : nump.array, shape=(dim_g,)
            Vector containing the number of events in observable space.
            If background was added the returned vector is A * vec_f + vec_b.

        vec_f : nump.array, shape=(dim_f,)
            Vector used to evaluate A * vec_f

        vec_f_reg : nump.array, shape=(dim_f,)
            Vector that should be passed to the regularization. For the
            BasisLinearModel it is identical to f.
        """
        vec_f = self.transform_vec_fit(vec_fit)
        vec_g, _, _ = super(BiasedLinearModel, self).evaluate(vec_f)
        return vec_g, vec_f, vec_fit

    def transform_vec_fit(self, vec_fit):
        """Transforms the fit vector to the actual vec_f which is e.g.
        used to evaluate the model.

        Parameters
        ----------
        vec_fit : np.array, shape=(dim_f)
            Vector which should be transformed into an acutal vec_f.


        Returns
        -------
        vec_f : np.array, shape=(dim_f)
            Vector in the space of the sought-after quantity.
        """
        eff_factor = self.model_factor_ - self.background_factor_
        vec_f = self.model_x0 * vec_fit * eff_factor
        return vec_f

    def generate_fit_x0(self, vec_g):
        """Generates a default seed for the minimization.
        The default seed vec_f_0 is a uniform distribution with
        sum(vec_f_0) = sum(vec_g). If background is present the default seed
        is: sum(vec_f_0) = sum(vec_g) - sum(self.vec_b).

        Parameters
        ----------
        vec_g : np.array, shape=(dim_g)
            Observable vector which should be used to get the correct
            normalization for vec_f_0.

        Returns
        -------
        vec_f_0 : np.array, shape=(dim_f)
            Seed vector of a minimization.
        """
        super(LinearModel, self).generate_fit_x0()
        return np.ones(self.dim_f)

    def generate_fit_bounds(self, vec_g):
        """Generates a bounds for a minimization.
        The bounds are (0, sum(vec_g)) without background and
        (0, sum(vec_g - self.vec_b)) with background. The bounds are for
        each fit parameter/entry in f.

        Parameters
        ----------
        vec_g : np.array, shape=(dim_g)
            Observable vector which should be used to get the correct
            upper bound

        Returns
        -------
        bounds : list, shape=(dim_f)
            List of tuples with the bounds.
        """
        super(LinearModel, self).generate_fit_bounds()
        n = self.A.shape[1]
        if self.has_background:
            n_events = np.sum(vec_g) - np.sum(self.vec_b)
        else:
            n_events = np.sum(vec_g)
        bounds = [(0, n_events)] * n
        return bounds

    def set_model_x0(self, model_x0, vec_g):
        """Sets the model_x0. Also the vec_g for the unfolding is need,
        to get centralize the fit around 1.

        Parameters
        ----------
        model_x0 : np.array, shape=(dim_f)
            Distribtuion used as a bias. Internally it is nomalized to
            sum(model_x0) = 1.

        vec_g : np.array, shape=(dim_g)
            Observable vector which is used to get the fit centered around
            1.
        """
        super(LinearModel, self).set_model_x0()
        if len(model_x0) != self.dim_f:
            raise ValueError("'model_x0' has to be of the length as "
                             "vec_f!")
        self.model_factor = sum(vec_g)
        self.model_x0 = model_x0 / self.model_factor

    def add_background(self, vec_b):
        """Adds a background vector to the model.

        Parameters
        ----------
        vec_b : numpy.array, shape=(dim_g)
            Vector g which is added to the model evaluation.
        """
        super(LinearModel, self).add_background()
        self.vec_b = vec_b
        self.background_factor = np.sum(vec_b)


class PolynominalSytematic(object):
    def __init__(self,
                 name,
                 degree,
                 prior=None,
                 bounds=None):
        self.name = name
        self.degree = degree
        if bounds is None:
            self.bounds = lambda x: True
        elif len(bounds) == 2:
            scale = bounds[1] - bounds[0]
            uniform_prior = stats.uniform(loc=bounds[0], scale=scale)
            self.bounds = lambda x: uniform_prior.pdf(x) > 0
        else:
            raise ValueError('bounds can be None or array-type with length 2')
        self.x = None
        self.coeffs = None
        if prior is None:
            def prior_pdf(x):
                return 1.

        elif hasattr(prior, 'pdf'):
            prior_pdf = prior.pdf
        elif callable(prior):
            prior_pdf = prior
        else:
            raise TypeError('The provided prior has to be None, '
                            'scipy.stats frozen rv or callable!')
        self.prior = prior
        self.prior_pdf = prior_pdf

    def lnprob_prior(self, x):
        if self.bounds(x):
            return np.log(self.prior_pdf(x))
        else:
            return np.inf * -1

    def sample_from_prior(self, size, sample_func_name=None):
        if hasattr(self.prior, 'rvs'):
            if self.bounds is None:
                samples = self.prior.rvs(size)
            else:
                samples = np.zeros(size, dtype=float)
                pointer = 0
                while pointer < size:
                    r = self.prior.rvs()
                    if self.bounds(r):
                        samples[pointer] = r
                        pointer += 1
        elif sample_func_name is not None:
            f = getattr(self.prior, sample_func_name)
            samples = f(size)
        else:
            raise TypeError(
                'Provided prior has neither a function called \'rvs\' nor '
                '\'sample_func_name\' was passed to the function!')
        return samples

    def add_data(self,
                 x,
                 baseline_idx,
                 digitized_obs,
                 sample_weights=None,
                 minlength_vec_g=0):
        self.baseline_idx = baseline_idx
        self.x = x
        if len(digitized_obs) != len(x):
            raise ValueError('digitized_obs has invalid shape! It needs to '
                             'be of shape (n_events, len(x))!')
        if sample_weights is not None:
            if len(sample_weights) != len(x):
                raise ValueError(
                    'digitized_obs has invalid shape! It needs to '
                    'be of shape (n_events, len(x))!')
        else:
            sample_weights = [None] * len(x)
        vectors_g = []
        for y_i, w_i in zip(digitized_obs, sample_weights):
            vectors_g.append(np.bincount(y_i,
                                         weights=w_i,
                                         minlength=minlength_vec_g))
        n_bins = np.unique(len(g) for g in vectors_g)
        if len(n_bins) > 1:
            raise ValueError(
                'digitized_obs has different number of populated bins! '
                'Either use different/same binning for all dataset or '
                'set minlength_vec_g')
        else:
            n_bins = n_bins[0]
        vectors_g = np.atleast_2d(vectors_g).T
        for i in range(len(x)):
            if i == baseline_idx:
                continue
            else:
                vectors_g[:, i] /= vectors_g[:, baseline_idx]
        vectors_g[:, baseline_idx] = 1.
        self.coeffs = np.empty((len(vectors_g), self.degree + 1), dtype=float)
        for i, y in enumerate(vectors_g):
            c = np.polyfit(x, y, self.degree)
            self.coeffs[i, :] = c

    def evaluate(self, baseline_digitized, x):
        factors = self.get_bin_factors(x)
        return factors[baseline_digitized]

    def get_bin_factors(self, x):
        if not self.bounds(x):
            return None
        factors = np.zeros(self.coeffs.shape[0], dtype=float)
        for i in range(self.degree + 1)[::-1]:
            factors += x**i * self.coeffs[:, i]
        return factors

    def __call__(self, baseline_digitized, x):
        return self.evaluate(baseline_digitized, x)


class ArrayCacheTransformation(object):
    def __init__(self, array):
        self.array = array

    def __call__(self, x):
        return self.array[np.argmin(np.absolute(x - self.array))]


class FloatCacheTransformation(object):
    def __init__(self, value, offset=0.):
        self.value = value
        self.offset = offset

    def __call__(self, x):
        a = (x - self.offset) / self.value
        return np.floor((a + 0.5)) * self.value + self.offset


class LinearModelSystematics(LinearModel):
    name = 'LinearModelSystematics'
    status_need_for_eval = 0

    def __init__(self, systematics=[], cache_precision=[]):
        super(LinearModelSystematics, self).__init__()
        self.range_obs = None
        self.range_truth = None
        self.A = None
        self.dim_f = None
        self.dim_g = None
        self.vec_b = None
        if systematics is None:
            systematics = []
        self.systematics = systematics
        if isinstance(cache_precision, list) or \
                isinstance(cache_precision, tuple):
            if len(cache_precision) == 0:
                cache_precision = None
        if cache_precision is None and len(self.systematics) > 0:
            cache_precision = [None] * len(self.systematics)
        self.cache_precision = cache_precision
        if cache_precision is not None:
            self.__cache = {}
            cache_error = ValueError('cache_precision has to be either None, '
                                     'float, (float, float) or a np.array!')
            for i, (s, p) in enumerate(zip(systematics, cache_precision)):
                if p is not None:
                    if isinstance(p, float):
                        self.cache_precision[i] = FloatCacheTransformation(p)
                    elif isinstance(p, list) or isinstance(p, tuple):
                        if len(p) == 2:
                            self.cache_precision[i] = FloatCacheTransformation(
                                value=p[0],
                                offset=p[1])
                        else:
                            raise cache_error
                    elif isinstance(p, np.array):
                        self.cache_precision[i] = ArrayCacheTransformation(p)
                    else:
                        raise cache_error
                    self.__cache[s.name] = {}
        self.n_nuissance_parameters = len(self.systematics)
        self.dim_fit_vector = None

    def initialize(self,
                   digitized_obs,
                   digitized_truth,
                   sample_weight=None):
        super(LinearModel, self).initialize()
        self.range_obs = (min(digitized_obs), max(digitized_obs))
        self.range_truth = (min(digitized_truth), max(digitized_truth))
        self.dim_f = self.range_truth[1] - self.range_truth[0] + 1
        self.dim_g = self.range_obs[1] - self.range_obs[0] + 1
        self.digitized_obs = digitized_obs
        self.digitized_truth = digitized_truth
        self.sample_weight = sample_weight
        self._A_unnormed = self.__generate_matrix_A_unnormed()
        self.A = np.dot(self._A_unnormed,
                        np.diag(1 / np.sum(self._A_unnormed, axis=0)))
        self.dim_fit_vector = self.dim_f + self.n_nuissance_parameters

    def __generate_matrix_A_unnormed(self, weight_factors=None):
        if self.sample_weight is None:
            weights = weight_factors
        else:
            if weight_factors is not None:
                weights = self.sample_weight * weight_factors
            else:
                weights = self.sample_weight
        binning_g, binning_f = self.__generate_binning__()
        A_unnormed = np.histogram2d(x=self.digitized_obs,
                                    y=self.digitized_truth,
                                    bins=(binning_g, binning_f),
                                    weights=weights)[0]
        return A_unnormed

    def evaluate_old(self, vec_fit):
        vec_f = vec_fit[:self.dim_f]
        nuissance_parameters = vec_fit[self.dim_f:]
        A = self._A_unnormed.copy()
        for s, x_s, c_t in zip(self.systematics,
                               nuissance_parameters,
                               self.cache_precision):
            factor_matrix = self.__get_systematic_event_factors(s, x_s, c_t)
            if factor_matrix is None:
                return np.array([-1.]), np.array([-1.]),  np.array([-1.])
            A *= factor_matrix
        M_norm = np.diag(1 / np.sum(A, axis=0))
        A = np.dot(A, M_norm)
        vec_g = np.dot(A, vec_f)
        if self.has_background:
            vec_g += self.vec_b
        return vec_g, vec_fit, vec_fit

    def evaluate(self, vec_fit):
        vec_f = vec_fit[:self.dim_f]
        nuissance_parameters = vec_fit[self.dim_f:]
        A = self._A_unnormed.copy()
        for s, x_s, c_t in zip(self.systematics,
                               nuissance_parameters,
                               self.cache_precision):
            factor_vector = self.__get_systematic_factors(s, x_s, c_t)
            if factor_vector is None:
                return np.array([-1.]), np.array([-1.]),  np.array([-1.])
            A *= factor_vector[:, np.newaxis]
        M_norm = np.diag(1 / np.sum(A, axis=0))
        A = np.dot(A, M_norm)
        vec_g = np.dot(A, vec_f)
        if self.has_background:
            vec_g += self.vec_b
        return vec_g, vec_fit, vec_fit

    def __get_systematic_event_factors(self,
                                       systematic,
                                       x,
                                       cache_transformation):
        if cache_transformation is not None:
            x = cache_transformation(x)
            if x in self.__cache[systematic.name].keys():
                return self.__cache[systematic.name][x]
        weight_factors = systematic(baseline_digitized=self.digitized_obs,
                                    x=x)
        if weight_factors is None:
            return None
        A_syst = self.__generate_matrix_A_unnormed(
            weight_factors=weight_factors)
        A_syst[A_syst > 0] /= self._A_unnormed[A_syst > 0]
        if cache_transformation is not None:
            self.__cache[systematic.name][x] = A_syst
        return A_syst

    def __get_systematic_factors(self, systematic, x, cache_transformation):
        if cache_transformation is not None:
            x = cache_transformation(x)
            if x in self.__cache[systematic.name].keys():
                return self.__cache[systematic.name][x]
        weight_factors = systematic.get_bin_factors(x=x)
        if weight_factors is None:
            return None
        if cache_transformation is not None:
            self.__cache[systematic.name][x] = weight_factors
        return weight_factors

    def generate_fit_x0(self, vec_g):
        vec_f_0 = super(LinearModelSystematics, self).generate_fit_x0(vec_g)
        vec_x_0 = np.ones(self.dim_fit_vector)
        vec_x_0[:self.dim_f] = vec_f_0

        for i, syst_i in enumerate(self.systematics):
            vec_x_0[self.dim_f + i] = syst_i.x[syst_i.baseline_idx]
        return vec_x_0

    def generate_fit_bounds(self, vec_g):
        bounds = super(LinearModelSystematics, self).generate_fit_bounds()
        for i, syst_i in enumerate(self.systematics):
            bounds.append(syst_i.bounds)
        return bounds

    def evaluate_condition(self, nuissance_parameters=None, normalize=True):
        """Returns an ordered array of the singular values of matrix A.

        Parameters
        ----------
        normalize : boolean (optional)
            If True the singular values return relativ to the largest
            value.

        Returns
        -------
        S_values : np.array, shape=(dim_f)
            Ordered array of the singular values.
        """
        if nuissance_parameters is not None:
            A = self.self._A_unnormed.copy()
            for s, x_s, c_t in zip(self.systematics,
                                   nuissance_parameters,
                                   self.cache_precision):
                factor_matrix = self.__get_systematic_factor(s, x_s, c_t)
                if factor_matrix is None:
                    return -1., -1., -1.
                A *= factor_matrix
            M_norm = np.diag(1 / np.sum(A, axis=0))
            A = np.dot(A, M_norm)
        else:
            A = self.A
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        U, S_values, V = linalg.svd(A)
        if normalize:
            S_values = S_values / S_values[0]
        return S_values
