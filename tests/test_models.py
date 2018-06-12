import numpy as np


def get_A_migrate_expample(shape, std=1, n_samples=1e4):
    assert shape[0] >= shape[1]
    y_binning = np.arange(shape[1] + 1) + 0.5
    A = np.ones(shape)
    mean_func = lambda i:  1 + (shape[1] - 1) / (shape[0] - 1) * (i)
    for i in range(A.shape[0]):
        mean = mean_func(i)
        random_sample = np.random.normal(loc=mean,
                                         scale=std,
                                         size=int(n_samples))
        A[i, :] = np.histogram(random_sample, y_binning)[0]
    return norm_A(A)


def norm_A(A):
    M_norm = np.diag(1 / np.sum(A, axis=0))
    return np.dot(A, M_norm)


if __name__ == '__main__':
    get_A_migrate_expample((20, 10))
