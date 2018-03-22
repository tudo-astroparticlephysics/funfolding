import numpy as np
from scipy.stats import norm


def calc_feldman_cousins_errors(best_fit,
                                sample,
                                sigma=1.):
    interval = int(sample.shape[0] * (norm.cdf(sigma) - norm.cdf(-sigma))) + 1
    diff = np.absolute(sample - best_fit)
    sigma_vec_best = np.zeros((2, len(best_fit)))
    for i in range(len(best_fit)):
        order = np.argsort(diff[:, i])
        select = order[:interval]
        selected_sample = sample[select, i]
        sigma_vec_best[0, i] = np.min(selected_sample, axis=0)
        sigma_vec_best[1, i] = np.max(selected_sample, axis=0)
    return sigma_vec_best


def calc_feldman_cousins_errors_binned(best_fit,
                                       sample,
                                       sigma=1.,
                                       eps=1e-2,
                                       percentiles=[10, 90]):
    interval = norm.cdf(sigma) - norm.cdf(-sigma)
    sigma_vec_best = np.zeros((2, len(best_fit)))
    for i, best_fit_i in enumerate(best_fit):
        sample_i = sample[:, i]
        percentile_i = np.percentile(sample_i, percentiles)

        range_percent = percentile_i[1] - percentile_i[0]
        eps_i = range_percent * eps
        min_s = np.min(sample_i)
        max_s = np.max(sample_i)
        print(eps_i)
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


def calc_errors_llh(sample, probs, sigma=1.):
    interval = int(len(probs) * (norm.cdf(sigma) - norm.cdf(-sigma))) + 1
    order = np.argsort(probs)
    selected = sample[np.sort(order[:interval]), :]
    sigma_vec_best = np.zeros((2, sample.shape[1]))
    sigma_vec_best[0, :] = np.max(selected, axis=1)
    sigma_vec_best[1, :] = np.min(selected, axis=1)
    return sigma_vec_best
