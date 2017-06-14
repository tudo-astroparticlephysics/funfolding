import numpy as np


def calc_feldman_cousins_errors(best_fit, sample, interval=0.68):
    diff = np.absolute(sample - best_fit)
    sample_length = sample.shape[0]
    order = np.argsort(diff)
    n_events = int(sample_length * interval) + 1
    select = order[:n_events]
    selected_sample = sample[select, :]
    sigma_vec_best = np.zeros((2, len(best_fit)))

    sigma_vec_best[0, :] = np.min(selected_sample, axis=0)
    sigma_vec_best[1, :] = np.max(selected_sample, axis=0)

    return sigma_vec_best
