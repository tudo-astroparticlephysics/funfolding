import warnings
import numpy as np
from scipy import linalg
from scipy import stats
from numpy.linalg import svd
import six


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
    def __init__(self, random_state=None):
        super(LinearModel, self).__init__()
        self.range_obs = None
        self.range_truth = None
        self.A = None
        self.dim_f = None
        self.dim_g = None
        self.vec_b = None
        self.dim_fit_vector = None
        self.x0_distributions = None
        self.n_nuissance_parameters = 0
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

    def initialize(self, digitized_obs, digitized_truth, sample_weight=None):
        """"""
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
        self.x0_distributions = [('poisson', None, 1)] * self.dim_f

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

    def generate_fit_x0(self, vec_g, vec_f_0, size=None):
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
        n = self.dim_f
        if vec_f_0 is None:

            if self.has_background:
                vec_f_0 = np.ones(n) * (np.sum(vec_g) - np.sum(self.vec_b)) / n
            else:
                vec_f_0 = np.ones(n) * np.sum(vec_g) / n
        if size is None:
            return vec_f_0
        pos_x0 = np.ones((size, n), dtype=float)
        x0_pointer = 0
        for (sample_x0, _, n_parameters) in self.x0_distributions[:self.dim_f]:
            if n_parameters == 1:
                x0_slice = x0_pointer
            else:
                x0_slice = slice(x0_pointer, x0_pointer + n_parameters)
            x0_i = vec_f_0[x0_slice]
            if sample_x0 == 'poisson':
                pos_x0_i = self.random_state.poisson(x0_i,
                                                     size=size)
            else:
                raise ValueError(
                    'Only "poisson" as name for x0 sample'
                    'dist is implemented')
            pos_x0[:, x0_slice] = pos_x0_i
            x0_pointer += 1
        #  wiggle on each point
        wiggle = np.absolute(self.random_state.normal(size=pos_x0.shape))
        pos_x0 += wiggle
        return pos_x0

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


class PolynominalSytematic(object):
    n_parameters = 1

    def __init__(self,
                 name,
                 degree,
                 prior=None,
                 use_stat_error=True,
                 bounds=None):
        self.name = name
        self.degree = degree
        self.use_stat_error = use_stat_error
        if bounds is None:
            self.bounds = lambda x: True
            self._bounds = None
        elif len(bounds) == 2:
            scale = bounds[1] - bounds[0]
            uniform_prior = stats.uniform(loc=bounds[0], scale=scale)
            self.bounds = lambda x: uniform_prior.pdf(x) > 0
            self._bounds = bounds
        else:
            raise ValueError('bounds can be None or array-type with length 2')
        self.x = None
        self.coeffs = None
        if prior is None:
            def prior_pdf(x):
                return 1.

        elif hasattr(prior, 'pdf'):
            def prior_pdf(x):
                return sum(prior.pdf(x))
        elif callable(prior):
            def prior_pdf(x):
                return sum(prior(x))
        else:
            raise TypeError('The provided prior has to be None, '
                            'scipy.stats frozen rv or callable!')
        self.prior = prior
        self.prior_pdf = prior_pdf
        self.baseline_value = None

    def lnprob_prior(self, x):
        if self.bounds(x):
            pdf_val = self.prior_pdf(x)
            if pdf_val > 0.:
                return np.log(pdf_val)
            else:
                return np.inf * -1
        else:
            return np.inf * -1

    def sample(self, size, sample_func_name=None):
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
        x = np.atleast_1d(x)
        self.baseline_idx = baseline_idx
        self.baseline_value = x[baseline_idx]
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
        vector_g = []
        rel_uncert = []
        mean_w = None
        for y_i, w_i in zip(digitized_obs, sample_weights):
            if w_i is not None:
                if mean_w is None:
                    mean_w = np.mean(sample_weights[baseline_idx])
                w_i = w_i / mean_w
            vector_g.append(np.bincount(y_i,
                                        weights=w_i,
                                        minlength=minlength_vec_g))
            rel_uncert.append(np.sqrt(np.bincount(y_i,
                                      weights=w_i**2,
                                      minlength=minlength_vec_g)))
        del digitized_obs
        del sample_weights
        n_bins = np.unique(len(g) for g in vector_g)
        if len(n_bins) > 1:
            raise ValueError(
                'digitized_obs has different number of populated bins! '
                'Either use different/same binning for all dataset or '
                'set minlength_vec_g')
        else:
            n_bins = n_bins[0]
        vector_g = np.atleast_2d(vector_g).T
        rel_uncert = np.atleast_2d(rel_uncert).T
        rel_uncert /= vector_g
        for i in range(len(x)):
            if i == baseline_idx:
                continue
            else:
                vector_g[:, i] /= vector_g[:, baseline_idx]
        vector_g[:, baseline_idx] = 1.
        self.coeffs = np.empty((len(vector_g), self.degree + 1), dtype=float)
        for i, (y, uncert) in enumerate(zip(vector_g, rel_uncert)):
            if self.use_stat_error:
                c = np.polyfit(x, y, self.degree, w=1. / (uncert * y))
            else:
                c = np.polyfit(x, y, self.degree)
            self.coeffs[i, :] = c

        self.vector_g = vector_g
        self.rel_uncert = rel_uncert
        self.x = x

    def plot(self, bin_i):
        from matplotlib import pyplot as plt
        if self.coeffs is None:
            raise RuntimeError("No data added yet. Call 'add_data' first.")

        fig, ax = plt.subplots()
        x_lim = [min(self.x), max(self.x)]
        x_lim[0] = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.1
        x_lim[1] = x_lim[1] + (x_lim[1] - x_lim[0]) * 0.1
        if self._bounds is not None:
            x_lim = self._bounds
        ax.set_xlim(x_lim)

        x_points = np.linspace(x_lim[0], x_lim[1], 100)
        y_points = np.zeros_like(x_points)
        for i in range(self.degree + 1)[::-1]:
            coeff_pointer = self.coeffs.shape[1] - 1 - i
            y_points += x_points**i * self.coeffs[bin_i, coeff_pointer]
        ax.plot(x_points, y_points, '-', color='0.5')

        yerr = self.rel_uncert[bin_i] * self.vector_g[bin_i]

        y_min = np.min(self.vector_g[bin_i] - yerr)
        y_max = np.max(self.vector_g[bin_i] + yerr)
        offset = (y_max - y_min) * 0.05

        ax.set_ylim(y_min - offset, y_max + offset)

        ax.errorbar(np.array(self.x),
                    self.vector_g[bin_i],
                    yerr=yerr,
                    fmt='o',
                    color='b')
        return fig, ax

    def evaluate(self, baseline_digitized, x):
        factors = self.get_bin_factors(x)
        return factors[baseline_digitized]

    def get_bin_factors(self, x):
        if not self.bounds(x):
            return None
        factors = np.zeros(self.coeffs.shape[0], dtype=float)
        for i in range(self.degree + 1)[::-1]:
            coeff_pointer = self.coeffs.shape[1] - 1 - i
            factors += x**i * self.coeffs[:, coeff_pointer]
        return factors

    def __call__(self, baseline_digitized, x):
        return self.evaluate(baseline_digitized, x)


def plane_fit(points):
    """
    https://stackoverflow.com/a/18968498
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = np.reshape(points, (np.shape(points)[0], -1))
    assert points.shape[0] <= points.shape[1], \
        "There are only {} points in {} dimensions.".format(points.shape[1],
                                                            points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:, -1]


def plane_fit_least_squares(points):
    A = np.ones((points.shape[0], 3), dtype=float)
    A[:, :2] = points[:, :2]
    A = np.matrix(A)
    b = np.matrix(points[:, 2]).T
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    return fit, errors


class CircularSystematic(object):
    n_parameters = 1

    def __init__(self,
                 name,
                 prior=None,
                 bounds=None):
        self.name = name
        if bounds is None:
            self.bounds = lambda x: True
            self._bounds = None
        elif len(bounds) == 2:
            scale = bounds[1] - bounds[0]
            uniform_prior = stats.uniform(loc=bounds[0], scale=scale)
            self.bounds = lambda x: uniform_prior.pdf(x) > 0
            self._bounds = bounds
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
        self.baseline_value = None

    def lnprob_prior(self, x):
        if self.bounds(x):
            pdf_val = self.prior_pdf(x)
            if pdf_val > 0.:
                return np.log(pdf_val)
            else:
                return np.inf * -1
        else:
            return np.inf * -1

    def sample(self, size, sample_func_name=None):
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
                 baseline_idx,
                 digitized_obs,
                 sample_weights=None,
                 minlength_vec_g=0):
        n_points = len(digitized_obs)
        x = np.linspace(0., 360., n_points, endpoint=True)

        self.baseline_idx = baseline_idx
        self.baseline_value = x[baseline_idx]
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
        vector_g = []
        rel_uncert = []
        mean_w = None
        for y_i, w_i in zip(digitized_obs, sample_weights):
            if w_i is not None:
                if mean_w is None:
                    mean_w = np.mean(sample_weights[baseline_idx])
                w_i = w_i / mean_w
            vector_g.append(np.bincount(y_i,
                                        weights=w_i,
                                        minlength=minlength_vec_g))
            rel_uncert.append(np.sqrt(np.bincount(y_i,
                                      weights=w_i**2,
                                      minlength=minlength_vec_g)))
        del digitized_obs
        del sample_weights
        n_bins = np.unique(len(g) for g in vector_g)
        if len(n_bins) > 1:
            raise ValueError(
                'digitized_obs has different number of populated bins! '
                'Either use different/same binning for all dataset or '
                'set minlength_vec_g')
        else:
            n_bins = n_bins[0]
        vector_g = np.atleast_2d(vector_g).T
        rel_uncert = np.atleast_2d(rel_uncert).T
        rel_uncert /= vector_g
        for i in range(len(x)):
            if i == baseline_idx:
                continue
            else:
                vector_g[:, i] /= vector_g[:, baseline_idx]
        vector_g[:, baseline_idx] = 1.
        self.vector_g = vector_g
        self.rel_uncert = rel_uncert
        self.x = x
        __x = np.zeros(len(x) + 1, dtype=float)
        __x[:-1] = x
        __x[-1] = __x[-2] + np.spacing(__x[-2])
        self.__x = __x
        self.distance = self.x[1] - self.x[0]

    def plot(self, bin_i):
        raise NotImplementedError

    def evaluate(self, baseline_digitized, x):
        factors = self.get_bin_factors(x)
        return factors[baseline_digitized]

    def get_bin_factors(self, x):
        if not self.bounds(x):
            return None
        x = x % 360.
        distance = np.sort(np.argmin(np.absolute(self.__x - x))[:2])
        idx_front = distance[1]
        idx_back = distance[0]
        if idx_front == len(x) - 1:
            idx_front = 0
        factor = self.x[idx_back] / self.distance
        contribution_back = factor * self.vector_g[idx_back]
        contribution_front = (1. - factor) * self.vector_g[idx_front]
        return contribution_back + contribution_front

    def __call__(self, baseline_digitized, x):
        return self.evaluate(baseline_digitized, x)


class PlaneSytematic(object):
    n_parameters = 2

    def __init__(self,
                 name,
                 prior=None,
                 bounds=None):
        self.name = name
        if bounds is None:
            self.bounds = lambda x: True
            self._bounds = None
        elif len(bounds) == 2:
            bounds_x = bounds[0]
            bounds_y = bounds[1]
            if bounds_x is None and bounds_y is not None:
                uniform_prior = stats.uniform(
                    loc=bounds_y[0],
                    scale=bounds_y[1] - bounds_y[0])
                self.bounds = lambda x: uniform_prior.pdf(x[1]) > 0
                self._bounds = (None, bounds_y)
            elif bounds_x is not None and bounds_y is None:
                uniform_prior = stats.uniform(
                    loc=bounds_x[0],
                    scale=bounds_x[1] - bounds_x[0])
                self.bounds = lambda x: uniform_prior.pdf(x[0]) > 0
                self._bounds = (bounds_x, None)
            elif bounds_x is not None and bounds_y is not None:
                uniform_prior = stats.uniform(
                    loc=(bounds_x[0], bounds_y[0]),
                    scale=(bounds_x[1] - bounds_x[0],
                           bounds_y[1] - bounds_y[0]))
                self.bounds = lambda x: all(uniform_prior.pdf(x) > 0)
                self._bounds = bounds
            else:
                self.bounds = lambda x: True
                self._bounds = None
        else:
            raise ValueError(
                "'bounds' can be either None or a tuple/list of len 2 "
                " containing None or the acutal bounds for")
        self.points = None
        self.coeffs = None
        if prior is None:
            def prior_pdf(x):
                return 1.
        elif hasattr(prior, 'pdf'):
            def prior_pdf(x):
                return sum(prior.pdf(x))
        elif callable(prior):
            def prior_pdf(x):
                return sum(prior(x))
        else:
            raise TypeError('The provided prior has to be None, '
                            'scipy.stats frozen rv or callable!')
        self.prior = prior
        self.prior_pdf = prior_pdf
        self.baseline_value = None

    def lnprob_prior(self, x):
        if self.bounds(x):
            p_val = self.prior_pdf(x)
            if p_val > 0.:
                return np.inf * -1
            else:
                return np.log(p_val)
        else:
            return np.inf * -1

    def sample(self, size, sample_func_name=None):
        if hasattr(self.prior, 'rvs'):
            if self.bounds is None:
                samples = self.prior.rvs(size)
            else:
                samples = np.zeros((size, 2), dtype=float)
                pointer = 0
                while pointer < size:
                    r = self.prior.rvs(size=(1, 2))
                    if self.bounds(r):
                        samples[pointer, :] = r
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
                 xy_coords,
                 baseline_idx,
                 digitized_obs,
                 sample_weights=None,
                 minlength_vec_g=0):
        self.baseline_idx = baseline_idx
        xy_coords = np.atleast_2d(xy_coords)
        self.baseline_value = xy_coords[baseline_idx, :]
        if len(digitized_obs) != len(xy_coords):
            raise ValueError('digitized_obs has invalid shape! It needs to '
                             'be of shape (n_events, len(x))!')
        if sample_weights is not None:
            if len(sample_weights) != len(xy_coords):
                raise ValueError(
                    'digitized_obs has invalid shape! It needs to '
                    'be of shape (n_events, len(x))!')
        else:
            sample_weights = [None] * len(xy_coords)
        vector_g = []
        mean_w = None
        for y_i, w_i in zip(digitized_obs, sample_weights):
            if w_i is not None:
                if mean_w is None:
                    mean_w = np.mean(sample_weights[baseline_idx])
                w_i /= mean_w
            vector_g.append(np.bincount(y_i,
                                        weights=w_i,
                                        minlength=minlength_vec_g))
        n_bins = np.unique(len(g) for g in vector_g)
        if len(n_bins) > 1:
            raise ValueError(
                'digitized_obs has different number of populated bins! '
                'Either use different/same binning for all dataset or '
                'set minlength_vec_g')
        else:
            n_bins = n_bins[0]
        vector_g = np.atleast_2d(vector_g).T
        for i in range(len(xy_coords)):
            if i == baseline_idx:
                continue
            else:
                vector_g[:, i] /= vector_g[:, baseline_idx]
        vector_g[:, baseline_idx] = 1.

        points = np.zeros((vector_g.shape[0],
                           xy_coords.shape[0],
                           xy_coords.shape[1] + 1), dtype=float)
        points[:, :, :2] = xy_coords
        points[:, :, 2] = vector_g
        self.coeffs = np.empty((vector_g.shape[0], xy_coords.shape[1] + 1))
        for i in range(vector_g.shape[0]):
            fit_i, _ = plane_fit_least_squares(points[i, :, :])
            self.coeffs[i, :] = fit_i.flatten()
        self.points = points

    def plot(self, bin_i):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        if self.coeffs is None:
            raise RuntimeError("No data added yet. Call 'add_data' first.")
        points = self.points[bin_i, :, :]
        coeffs = self.coeffs[bin_i, :]

        x_lim = [np.min(points[:, 0], axis=0), np.max(points[:, 0], axis=0)]
        y_lim = [np.min(points[:, 1], axis=0), np.max(points[:, 1], axis=0)]
        x_lim[0] = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.1
        x_lim[1] = x_lim[1] + (x_lim[1] - x_lim[0]) * 0.1
        y_lim[0] = y_lim[0] - (y_lim[1] - y_lim[0]) * 0.1
        y_lim[1] = y_lim[1] + (y_lim[1] - y_lim[0]) * 0.1
        if self._bounds is not None:
            x_lim_bounds = self._bounds[0]
            if x_lim_bounds is not None:
                x_lim = x_lim_bounds
            y_lim_bounds = self._bounds[1]
            if y_lim_bounds is not None:
                y_lim = y_lim_bounds
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_xticks(np.unique(points[:, 0]))
        ax.set_yticks(np.unique(points[:, 1]))
        X, Y = np.meshgrid(np.arange(x_lim[0], x_lim[1], 0.02),
                           np.arange(y_lim[0], y_lim[1], 0.02))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r, c] = coeffs[0] * X[r, c] + coeffs[1] * Y[r, c] + coeffs[2]
        idx = np.ones(len(points), dtype=bool)
        idx[self.baseline_idx] = False
        z_min, z_max = np.min(Z), np.max(Z)
        diff = z_max - z_min
        ax.set_zlim(z_min - diff * 0.05, z_max + diff * 0.05)
        for i, (x, y, z) in enumerate(points):
            if i == self.baseline_idx:
                color = 'r'
            else:
                color = 'k'
            ax.plot([x, x],
                    [y, y],
                    [ax.get_zlim()[0], z],
                    '-*',
                    color=color)
        ax.plot_wireframe(X, Y, Z, color='C0')
        return fig, ax

    def evaluate(self, baseline_digitized, x):
        factors = self.get_bin_factors(x)
        return factors[baseline_digitized]

    def get_bin_factors(self, x):
        if not self.bounds(x):
            return None
        __x = np.ones(len(x) + 1, dtype=float)
        __x[:len(x)] = x
        return np.sum(self.coeffs * __x, axis=1)

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

    def __init__(self,
                 generic_epsilon=None,
                 systematics=[],
                 cache_precision=[],
                 random_state=None):
        super(LinearModelSystematics, self).__init__(random_state=random_state)
        self.range_obs = None
        self.range_truth = None
        self.A = None
        self.dim_f = None
        self.dim_g = None
        self.vec_b = None
        if systematics is None:
            systematics = []
        if cache_precision is None:
            cache_precision = []
        self.systematics = systematics
        if len(cache_precision) < len(self.systematics):
            for i in range(len(self.systematics) - len(cache_precision)):
                cache_precision.append(None)
        elif len(cache_precision) > len(self.systematics):
            raise ValueError('len(systematics) should be len(cache_precision')
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
        self.n_nuissance_parameters = sum(s.n_parameters
                                          for s in self.systematics)
        self.dim_fit_vector = None
        self.x0_distributions = None
        if generic_epsilon is not None:
            if isinstance(generic_epsilon, float):
                if generic_epsilon <= 0.:
                    raise ValueError('generic_epsilon has to be > 0.')
                else:
                    self.n_nuissance_parameters += 1
            else:
                raise ValueError('generic_epsilon has to be None or float > 0')
        self.generic_epsilon = generic_epsilon

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
        self.x0_distributions = [('poisson', None, 1)] * self.dim_f
        self.x0_distributions += [(s.sample, s.lnprob_prior, s.n_parameters)
                                  for s in self.systematics]
        if self.generic_epsilon is not None:
            s = stats.norm(loc=1., scale=self.generic_epsilon)
            s.random_state = self.random_state

            def lnprop_prior_generic_epsilon(x):
                val = s.pdf(x)[0]
                if val == 0.:
                    return -np.inf
                else:
                    return np.log(val)

            self.x0_distributions += [(s.rvs, lnprop_prior_generic_epsilon, 1)]

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
                return np.array([-1.]), np.array([-1.]), np.array([-1.])
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
        pointer = 0
        for syst_i, c_t in zip(self.systematics, self.cache_precision):
            s = slice(pointer, pointer + syst_i.n_parameters)
            x_s = nuissance_parameters[s]
            factor_vector = self.__get_systematic_factors(syst_i, x_s, c_t)
            if factor_vector is None:
                return np.array([-1.]), np.array([-1.]), np.array([-1.])
            A *= factor_vector[:, np.newaxis]
            pointer += syst_i.n_parameters
        M_norm = np.diag(1 / np.sum(A, axis=0))
        A = np.dot(A, M_norm)
        if self.generic_epsilon is not None:
            vec_f *= vec_fit[-1]
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

    def generate_fit_x0(self, vec_g, vec_f_0=None, size=None):
        vec_f_0_def_f = super(LinearModelSystematics, self).generate_fit_x0(
            vec_g=vec_g,
            vec_f_0=vec_f_0,
            size=None,
        )
        vec_x_0_def = np.ones(self.dim_fit_vector, dtype=float)
        vec_x_0_def[:self.dim_f] = vec_f_0_def_f
        x0_pointer = self.dim_f
        for syst_i in self.systematics:
            s = slice(x0_pointer, x0_pointer + syst_i.n_parameters)
            vec_x_0_def[s] = syst_i.baseline_value
            x0_pointer += syst_i.n_parameters
        if size is None:
            return vec_x_0_def

        pos_x0 = np.ones((size, self.dim_fit_vector), dtype=float)
        vec_f_x0 = super(LinearModelSystematics, self).generate_fit_x0(
            vec_g=vec_g,
            vec_f_0=vec_f_0,
            size=size,
        )
        pos_x0[:, :self.dim_f] = vec_f_x0
        x0_pointer = self.dim_f
        for sample_x0, _, n_parameters in self.x0_distributions[self.dim_f:]:
            if n_parameters == 1:
                x0_slice = x0_pointer
            else:
                x0_slice = slice(x0_pointer, x0_pointer + n_parameters)
            x0_i = vec_x_0_def[x0_slice]
            if sample_x0 is None:
                pos_x0_i = x0_i
            elif isinstance(sample_x0, six.string_types):
                if sample_x0 == 'poisson':
                    pos_x0_i = self.random_state.poisson(x0_i,
                                                         size=size)
                else:
                    raise ValueError(
                        'Only "poisson" as name for x0 sample'
                        'dist is implemented')
            elif callable(sample_x0):
                pos_x0_i = sample_x0(size=size)
            pos_x0[:, x0_slice] = pos_x0_i
            x0_pointer += n_parameters
        if self.generic_epsilon is not None:
            for i, factor in enumerate(pos_x0[:, -1]):
                pos_x0[i, :self.dim_f] /= factor
        return pos_x0

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


class TestModelSystematics(LinearModelSystematics):
    name = 'TestModelSystematics'
    status_need_for_eval = 0

    def __init__(self,
                 systematics=[],
                 cache_precision=[]):
        super(LinearModelSystematics, self).__init__(
            systematics=systematics,
            cache_precision=cache_precision)
        self.f_test = None

    def initialize(self,
                   f_test,
                   digitized_obs,
                   digitized_truth,
                   sample_weight=None):
        super(TestModelSystematics, self).initialize(
            digitized_obs=digitized_truth,
            digitized_truth=digitized_truth,
            sample_weight=sample_weight)
        if len(f_test) == self.dim_f:
            self.f_test = f_test / np.sum(f_test)
        else:
            raise ValueError(
                '\'f_test\' wrong length! Has {} needs {} '.format(
                    len(f_test),
                    self.dim_f))
        self.dim_fit_vector = 1 + self.n_nuissance_parameters

    def evaluate(self, vec_fit):
        vec_fit_transformed = self.transform_fit_vector(vec_fit)
        return super(TestModelSystematics, self).evaluate(
            vec_fit=vec_fit_transformed)

    def generate_fit_x0(self, vec_g):
        factor = np.sum(vec_g) / np.sum(self.f_test)
        vec_x_0 = np.ones(self.dim_fit_vector)
        vec_x_0[0] = factor

        for i, syst_i in enumerate(self.systematics):
            vec_x_0[1 + i] = syst_i.x[syst_i.baseline_idx]
        return vec_x_0

    def generate_fit_bounds(self, vec_g, max_factor=3.):
        n_events = np.sum(vec_g)
        bounds = [(0., n_events * max_factor)]
        for i, syst_i in enumerate(self.systematics):
            bounds.append(syst_i.bounds)
        return bounds

    def transform_fit_vector(self, vec_fit):
        vec_f = self.f_test * vec_fit[0]
        vec_fit_transformed = np.zeros(
            self.self.dim_f + self.n_nuissance_parameters, dtype=float)
        vec_fit_transformed[:self.dim_f] = vec_f
        return vec_fit_transformed
