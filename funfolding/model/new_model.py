import logging
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt


class Model:
    name = 'Model'
    status_need_for_eval = 0
    """ Base class for a model. Actual models should inherit from this class.

    In this class the functions that should be implemented by each model are
    defined and some logging is implemented.

    Attributes
    ----------
    name : str
        Name of the model.

    logger : logging.Logger
        Instance of a Logger. The name of the logger is the name of the
        model.

    status : int
        Indicates the status of the model:
            -1 : Instance created. Not filled with values yet.
             0 : Filled with values
             1 : Filled with values and x0 set (optional level).
    """
    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.debug('Created {}'.format(self.name))
        self.status = -1

    def initialize(self):
        """This function should be called with all needed values. To actually
        fill all the models with values.
        """
        self.logger.debug('\tModel initialized!')
        self.status = 0

    def evaluate(self):
        """Evaluates the model.

        Actual implementation of this functions should return:
            g     : Observable vector
            f     : Solution vector
            f_reg : Vector used in the regularization

        """
        self.logger.debug('\tEvaluation!')
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
        self.logger.debug('\tSetting up model x0!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized, before setting x0. "
                               "Run 'model.initialize' first!")
        self.status = 1

    def generate_fit_x0(self):
        """The model should be able to return resonable starting values
        for the fitter.
        """
        self.logger.debug('\tGenerating x0 for the fitter!')
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
        self.logger.debug('\tGenerating bounds for the fitter!')
        if self.status < 0 and self.status_need_for_eval == 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        if self.status < 1 and self.status_need_for_eval == 1:
            raise RuntimeError("Model has to be intilized and x0 has to be"
                                "set. Run 'model.initialize' and "
                                "'model.set_x0' first!")


class BasicLinearModel(Model):
    name = 'BasicLinearModel'
    status_need_for_eval = 0
    """ Basic Linear model:
    g = A * f

    Attributes
    ----------
    name : str
        Name of the model.

    logger : logging.Logger
        Instance of a Logger. The name of the logger is the name of the
        model.

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
    """



    def __init__(self):
        super(BasicLinearModel, self).__init__()
        self.range_obs = None
        self.range_truth = None
        self.A = None
        self.dim_f = None
        self.dim_g = None

    def initialize(self, digitized_obs, digitized_truth, sample_weight=None):
        """

        """
        super(BasicLinearModel, self).initialize()
        self.range_obs = (min(digitized_obs), max(digitized_obs))
        self.range_truth = (min(digitized_truth), max(digitized_truth))
        self.dim_f = self.range_obs[1] - self.range_obs[0] + 1
        self.dim_g =  self.range_truth[1] - self.range_truth[0] + 1
        binning_g, binning_f = self.__generate_binning__()
        self.A = np.histogram2d(x=digitized_obs,
                                y=digitized_truth,
                                bins=(binning_g, binning_f),
                                weights=sample_weight)[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = np.dot(self.A, M_norm)

    def evaluate(self, f):
        super(BasicLinearModel, self).evaluate()
        return np.dot(self.A, f), f, f

    def generate_fit_x0(self, vec_g):
        super(BasicLinearModel, self).generate_fit_x0()
        n = self.A.shape[1]
        return np.ones(n) * np.sum(vec_g) / n

    def generate_fit_bounds(self, vec_g):
        super(BasicLinearModel, self).generate_fit_bounds()
        n = self.A.shape[1]
        n_events = np.sum(vec_g)
        bounds = []
        for i in range(n):
            bounds.append((0, n_events))
        return bounds

    def set_model_x0(self):
        super(BasicLinearModel, self).set_model_x0()
        self.logger.info('\tx0 has no effect for {}'.format(self.name))


    def evaluate_condition(self, ax=None, label='Linear Model'):
        self.logger.debug('Evaluation of Singular Values!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        U, S_values, V = linalg.svd(self.A)
        if ax is None:
            _, ax = plt.subplots()
        ax.set_xlabel(r'Index $j$')
        ax.set_ylabel(r'Normed Singular Values $\frac{\lambda_i}{\lambda_0}$')

        S_values = S_values / S_values[0]
        binning = np.linspace(-0.5,
                              len(S_values) - 0.5,
                              len(S_values) + 1)
        x_pos = np.arange(len(S_values))
        ax.hist(x_pos,
                bins=binning,
                weights=S_values,
                histtype='step',
                label=label)
        ax.set_xlim([binning[0], binning[-1]])
        return ax

    def __generate_binning__(self):
        self.logger.debug('\t\tGenerating binning vectors!')
        if self.status < 0:
            raise RuntimeError("Model has to be intilized. "
                               "Run 'model.initialize' first!")
        binning_obs = np.linspace(self.range_obs[0],
                                  self.range_obs[1] +1 ,
                                  self.dim_g + 1)
        binning_truth = np.linspace(self.range_truth[0],
                                  self.range_truth[1] +1 ,
                                  self.dim_f + 1)
        return binning_obs, binning_truth

    def generate_vectors(self, g=None, f=None):
        binning_g, binning_f = self.__generate_binning__()
        if g is not None:
            vec_g = np.histogram(g, bins=binning_g)[0]
        else:
            vec_g = None
        if f is not None:
            vec_f = np.histogram(f, bins=binning_f)[0]
        else:
            vec_f = None
        return vec_g, vec_f
