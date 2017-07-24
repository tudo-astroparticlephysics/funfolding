import warnings
import numpy as np
from scipy import linalg


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

    def generate_vectors(self, digitized_obs=None, digitized_truth=None):
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
            vec_g = np.histogram(digitized_obs, bins=binning_obs)[0]
        else:
            vec_g = None
        if digitized_truth is not None:
            vec_f = np.histogram(digitized_truth, bins=binning_truth)[0]
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
