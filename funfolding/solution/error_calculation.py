import numpy as np
from scipy.stats import norm


def calc_errors_llh(sample,
                    probs,
                    sigma=1.,
                    sigma_limits=None,
                    precision_f=0.1,
                    n_nuissance=0):
    if sigma_limits is None:
        sigma_limits = sigma
    if precision_f is not None:
        sample = sample.copy()
        for i in range(sample.shape[1] - n_nuissance):
            a = sample[:, i] / precision_f
            sample[:, i] = np.floor((a + 0.5)) * precision_f
    interval = int(len(probs) * (norm.cdf(sigma) - norm.cdf(-sigma))) + 1
    interval_limits = int(len(probs) * (norm.cdf(sigma_limits) -
                                        norm.cdf(-sigma_limits))) + 1
    order = np.argsort(probs)
    selected = sample[np.sort(order[:interval]), :]
    sigma_vec_best = np.zeros((2, sample.shape[1]))
    sigma_vec_best[0, :] = np.min(selected, axis=0)
    sigma_vec_best[1, :] = np.max(selected, axis=0)
    if sigma_limits != sigma:
        idx = np.sort(order[:interval_limits])
        selected = sample[idx, :]
        for i, calc_upper_limit in enumerate(sigma_vec_best[0, :] == 0.):
            if calc_upper_limit:
                sigma_vec_best[1, i] = np.max(selected[:, i])
    return sigma_vec_best


def calc_feldman_cousins_errors(best_fit,
                                sample,
                                sigma=1.,
                                sigma_limits=None,
                                precision_f=0.1,
                                n_nuissance=0):
    if sigma_limits is None:
        sigma_limits = sigma
    if precision_f is not None:
        sample = sample.copy()
        for i in range(sample.shape[1] - n_nuissance):
            a = sample[:, i] / precision_f
            sample[:, i] = np.floor((a + 0.5)) * precision_f
    interval = int(sample.shape[0] * (norm.cdf(sigma) - norm.cdf(-sigma))) + 1
    interval_limits = int(sample.shape[0] * (norm.cdf(sigma_limits) -
                                             norm.cdf(-sigma_limits))) + 1
    diff = np.absolute(sample - best_fit)
    sigma_vec_best = np.zeros((2, len(best_fit)))
    for i in range(len(best_fit)):
        order = np.argsort(diff[:, i])
        select = order[:interval]
        selected_sample = sample[select, i]
        sigma_vec_best[0, i] = np.min(selected_sample, axis=0)
        sigma_vec_best[1, i] = np.max(selected_sample, axis=0)
        if sigma_limits != sigma:
            if sigma_vec_best[0, i] == 0.:
                select = order[:interval_limits]
                selected_sample = sample[select, i]
                sigma_vec_best[1, i] = np.max(selected_sample, axis=0)
    return sigma_vec_best


def calc_feldman_cousins_errors_binned(best_fit,
                                       sample,
                                       sigma=1.,
                                       sigma_limits=None,
                                       precision_f=None,
                                       n_nuissance=0,
                                       eps=1e-2,
                                       percentiles=[10, 90]):
    interval = norm.cdf(sigma) - norm.cdf(-sigma)
    if precision_f is not None:
        sample = sample.copy()
        for i in range(sample.shape[1] - n_nuissance):
            a = sample[:, i] / precision_f
            sample[:, i] = np.floor((a + 0.5)) * precision_f
    interval_limits = norm.cdf(sigma_limits) - norm.cdf(-sigma_limits)
    sigma_vec_best = np.zeros((2, len(best_fit)))
    for i, best_fit_i in enumerate(best_fit):
        sample_i = sample[:, i]
        percentile_i = np.percentile(sample_i, percentiles)

        range_percent = percentile_i[1] - percentile_i[0]
        eps_i = range_percent * eps
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
        if np.min(added_bins) == 0. and sigma != sigma_limits:
            while sum_ < interval_limits:
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


def bayesian_parameter_estimation(sample,
                                  sigma=1.,
                                  sigma_limits=None,
                                  precision_f=0.1,
                                  n_nuissance=0):
    if sigma_limits is None:
        sigma_limits = sigma
    interval_lower = norm.cdf(-sigma) * 100.
    interval_upper = norm.cdf(sigma) * 100.
    upper_limit = norm.cdf(sigma_limits)
    spectrum = np.percentile(
        sample, q=[interval_lower, 50., interval_upper], axis=0
    )
    calc_limit = np.zeros(spectrum.shape[1], dtype=bool)
    calc_limit[:-n_nuissance] = spectrum[0, :-n_nuissance] < precision_f
    spectrum[0, calc_limit] = 0.
    spectrum[2, calc_limit] = np.percentile(sample[:, calc_limit],
                                            q=[upper_limit], axis=0)
    best_fit = spectrum[1, :]
    sigma_vec_best = np.zeros((2, len(best_fit)))
    sigma_vec_best[0, :] = spectrum[0, :]
    sigma_vec_best[1, :] = spectrum[2, :]
    return best_fit, sigma_vec_best
