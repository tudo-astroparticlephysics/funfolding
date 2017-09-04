import numpy as np


def calc_feldman_cousins_errors(best_fit, sample, interval=0.68):
    diff = np.absolute(sample - best_fit)
    sample_length = sample.shape[0]
    n_events = int(sample_length * interval) + 1
    sigma_vec_best = np.zeros((2, len(best_fit)))
    for i in range(len(best_fit)):
        order = np.argsort(diff[:, i])
        select = order[:n_events]
        selected_sample = sample[select, i]
        sigma_vec_best[0, i] = np.min(selected_sample, axis=0)
        sigma_vec_best[1, i] = np.max(selected_sample, axis=0)

    return sigma_vec_best


def calc_feldman_cousins_errors_binned(best_fit,
                                       sample,
                                       interval=0.68,
                                       eps=1e-2,
                                       percentiles=[10,90]):
    sigma_vec_best = np.zeros((2, len(best_fit)))
    for i, best_fit_i in enumerate(best_fit):
        sample_i = sample[:, i]
        percentile_i = np.percentile(sample_i, percentiles)
        range_percent = percentile_i[1] - percentile_i[0]
        eps_i = range_percent * eps
        print(eps_i)
        min_s = np.min(sample_i)
        max_s = np.max(sample_i)

        binning = np.arange(min_s - 0.5 * eps_i,
                            max_s + 1.5 * eps_i,
                            eps_i)
        hist = np.histogram(sample_i,
                            bins=binning)[0]
        hist = hist / float(sample.shape[0])
        bin_centers = (binning[1:] + binning[:-1]) / 2.
        diff = np.absolute(bin_centers - best_fit[i])
        order = np.argsort(diff)
        sum_ = 0.
        pointer = 0
        added_bins = []
        while sum_ < interval:
            idx = order[pointer]
            if idx > order[0]:
                added_bins.append(binning[idx + 1])
            else:
                added_bins.append(binning[idx])
            sum_ += hist[idx]
            pointer += 1
        sigma_vec_best[0, i] = np.min(added_bins)
        if sigma_vec_best[0, i] < 0.:
            sigma_vec_best[0, i] = 0.
        sigma_vec_best[1, i] = np.max(added_bins)
    return sigma_vec_best
