from funfolding.discretization import TreeBinning
import numpy as np
from sklearn.datasets import make_regression

from matplotlib import pyplot as plt

def test_treebinning():
    n_samples = 1000
    X, y = make_regression(n_samples=n_samples)
    plt.hist(y)
    y_binned = np.digitize(y, np.linspace(-400, 400, 20))
    idx = int(0.9 * n_samples)
    X_test = X[idx:, :]
    X_train = X[:idx, :]
    y_train = y_binned[:idx]
    clf = TreeBinning(regression=False)
    clf.fit(X_train, y_train)
    score = clf.predict(X_test)
    assert len(score) == X_test.shape[0]
    leaves = clf.apply(X_test)
    assert len(leaves) == X_test.shape[0]


if __name__ == '__main__':
    test_treebinning()
