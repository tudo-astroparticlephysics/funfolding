from .model import Model

class LinearMatrixModel(Model):
    def __init__(self):
        super(LinearMatrixModel, self).__init__()

    def initialize(self, X, y, sample_weight=None):
        super(LinearMatrixModel, self).initialize()
        X_binning = np.unique(X)
        X_binning = np.vstack((X_binning -0.5, X_binning[-1] + 0.5))
        y_binning = np.unique(y)
        y_binning = np.vstack((y_binning -0.5, y_binning[-1] + 0.5))
        self.A = np.histogram2d(x=X,
                                y=y,
                                bins=(X_binning, y_binning),
                                weights=sample_weight)[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = np.dot(self.A, M_norm)

    def evaluation(self, f):
        super(LinearMatrixModel, self).evaluation()
        return np.dot(self.A, f)
