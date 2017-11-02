#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._binning import Binning

import itertools
import numpy as np

import copy

try:
    from astroML.density_estimation.bayesian_blocks import bayesian_blocks
    got_astroML = True
except ImportError:
    got_astroML = False


class ClassicBinning(Binning):
    name = 'ClassicalBinning'
    status_need_for_digitize = 0

    def __init__(self,
                 bins,
                 range=None,
                 oor_handle='individual',
                 random_state=None):
        super(ClassicBinning, self).__init__()
        self.hist_func = np.histogram
        self.n_dims = len(bins)
        self.bins = bins
        if range is None:
            self.range = [None] * self.n_dims
        else:
            self.range = range
        self.edges = []
        self.t_to_i = None
        self.i_to_t = None
        self.n_bins = None
        self.oor_tuples = None
        self.oor_handle = oor_handle
        if not isinstance(random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

    def initialize(self,
                   X,
                   y=None,
                   sample_weight=None):
        super(ClassicBinning, self).initialize()
        for dim_i in range(self.n_dims):
            if sample_weight is None:
                w_i = None
            else:
                w_i = sample_weight[:, dim_i]
            if self.bins[dim_i] == 'blocks':
                if got_astroML:
                    self.edges.append(bayesian_blocks(X[:, dim_i]))
                else:
                    raise RuntimeError("Install astroML to use 'blocks'")
            else:
                self.edges.append(self.hist_func(a=X[:, dim_i],
                                                 bins=self.bins[dim_i],
                                                 range=self.range[dim_i],
                                                 weights=w_i)[1])
        self.create_conversion_dict()
        self.n_bins = len(self.t_to_i.keys())

    def digitize(self, X, right=False):
        super(ClassicBinning, self).digitize()
        tup_label = np.zeros((len(X), self.n_dims), dtype=int)
        for dim_i in range(self.n_dims):
            digi = np.digitize(x=X[:, dim_i],
                               bins=self.edges[dim_i],
                               right=right)
            tup_label[:, dim_i] = digi
        return self.convert_tup_label(tup_label)

    def convert_tup_label(self, tup_label):
        i_label = np.array([self.t_to_i[tuple(key)] for key in tup_label],
                           dtype=int)
        return i_label

    def create_conversion_dict(self):
        range_list = [np.arange(len(edges_i) + 1)
                      for edges_i in self.edges]
        indices = itertools.product(*range_list)
        self.t_to_i = {x: i for i, x in enumerate(indices)}

        def is_oor(tup_i):
            lower = any(np.array(tup_i) == 0)
            upper = any([t == len(self.edges[i]) for i, t in enumerate(tup_i)])
            return lower or upper

        self.i_to_t = {self.t_to_i[t]: t for t in self.t_to_i.keys()}
        self.oor_tuples = set(t for t in self.t_to_i.keys() if is_oor(t))

    def copy(self):
        clone = ClassicBinning(bins=self.bins)
        clone.bins = copy.deepcopy(self.bins)
        clone.range = copy.deepcopy(self.range)
        clone.edges = copy.deepcopy(self.edges)
        clone.t_to_i = copy.deepcopy(self.t_to_i)
        clone.i_to_t = copy.deepcopy(self.i_to_t)
        clone.n_bins = copy.deepcopy(self.n_bins)
        clone.oor_tuples = copy.deepcopy(self.oor_tuples)
        clone.oor_handle = copy.deepcopy(self.oor_handle)
        clone.random_state = copy.deepcopy(self.random_state)
        clone.status = int(self.status)
        return clone

    def __merge__(self,
                  X,
                  min_samples=None,
                  max_bins=None,
                  sample_weight=None,
                  y=None,
                  right=False,
                  mode='closest',
                  **kwargs):
        super(ClassicBinning, self).merge()
        n_merg_iterations = 0
        binned = self.digitize(X, right=right)
        counted = np.bincount(binned,
                              weights=sample_weight,
                              minlength=self.n_bins)
        original_sum = np.sum(counted)

        if min_samples is None and max_bins is None:
            raise ValueError("Either 'min_samples' or 'max_bins' have "
                             "to be set!")
        elif min_samples is None:
            min_samples = original_sum
        elif max_bins is None:
            max_bins = 1
        if mode == 'similar':
            if y is None:
                raise ValueError("For mode 'similar' labels are needed!")
            if sample_weight is None:
                w = y
            else:
                w = y * sample_weight
            no_entry = counted == 0
            self.mean_label = np.bincount(binned,
                                          weights=w,
                                          minlength=self.n_bins)
            self.mean_label[no_entry] = np.nan
            self.mean_label /= counted
            self.__get_bin_for_merge__ = self.__get_most_similar_neighbor__
        elif mode == 'lowest':
            self.__get_bin_for_merge__ = self.__get_lowest_neighbor__
        elif mode == 'closest':
            self.__get_bin_for_merge__ = self.__get_closest_neighbor__
        else:
            raise ValueError("'closest', 'lowest' and 'similar' are "
                             "valid options for keyword 'mode'")
        while True:
            min_val = np.min(counted)
            try:
                assert min_val <= min_samples
                assert self.n_bins > max_bins
            except AssertionError:
                break
            else:
                min_indices = np.where(counted == min_val)[0]
                min_idx = self.random_state.choice(min_indices)
                neighbors = self.__get_neighbors__(min_idx)
                i_label_keep, i_label_remove = self.__get_bin_for_merge__(
                    bin_a=min_idx,
                    neighbors=neighbors,
                    counted=counted,
                    **kwargs)
                self.__merge_bins__(i_label_keep, i_label_remove)
                counted[i_label_keep] += counted[i_label_remove]
                mask = np.ones_like(counted, dtype=bool)
                mask[i_label_remove] = False
                counted = counted[mask]
                n_merg_iterations += 1
                self.n_bins -= 1
                if np.sum(counted) != original_sum:
                    raise RuntimeError('Events sum changed!')

        self.n_bins = len(self.i_to_t)
        return self

    def merge(self,
              X,
              min_samples=None,
              max_bins=None,
              sample_weight=None,
              y=None,
              right=False,
              mode='closest',
              inplace=False):
        if inplace:
            return self.__merge__(X=X,
                                  min_samples=min_samples,
                                  max_bins=max_bins,
                                  sample_weight=sample_weight,
                                  y=y,
                                  right=right,
                                  mode=mode)
        else:
            clone = self.copy()
            return clone.merge(X=X,
                               min_samples=min_samples,
                               max_bins=max_bins,
                               sample_weight=sample_weight,
                               y=y,
                               right=right,
                               mode=mode,
                               inplace=True)

    def __get_neighbors__(self, i_label):
        t_labels = self.i_to_t[i_label]
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        neighbors = set()
        for t_label in t_labels:
            dims = range(self.n_dims)
            for i in dims:
                upper = []
                lower = []
                for j in dims:
                    if j == i:
                        upper.append(t_label[j] + 1)
                        lower.append(t_label[j] - 1)
                    else:
                        upper.append(t_label[j])
                        lower.append(t_label[j])
                upper = tuple(upper)
                lower = tuple(lower)
                try:
                    if upper not in t_labels:
                        neighbors.add(self.t_to_i[upper])
                except KeyError:
                    pass
                try:
                    if lower not in t_labels:
                        neighbors.add(self.t_to_i[lower])
                except KeyError:
                    pass
        assert i_label not in neighbors
        return list(neighbors)

    def __merge_bins__(self, i_label_keep, i_label_remove):
        if i_label_remove <= i_label_keep:
            raise RuntimeError
        t_labels_remove = self.i_to_t[i_label_remove]
        if isinstance(t_labels_remove, tuple):
            t_labels_remove = [t_labels_remove]

        t_labels_keep = self.i_to_t[i_label_keep]
        if isinstance(t_labels_keep, tuple):
            t_labels_keep = [t_labels_keep]

        for t_label_remove in t_labels_remove:
            self.t_to_i[t_label_remove] = i_label_keep

        for t_label in self.t_to_i.keys():
            if self.t_to_i[t_label] > i_label_remove:
                self.t_to_i[t_label] -= 1
        self.i_to_t = {}
        for t_label, i_label in self.t_to_i.items():
            t_labels = self.i_to_t.get(i_label, [])
            t_labels.append(t_label)
            self.i_to_t[i_label] = t_labels
        return i_label_keep, i_label_remove

    def __get_lowest_neighbor__(self, bin_a, neighbors, counted):
        counted_neighbors = counted[neighbors]
        min_val = np.where(counted_neighbors == np.min(counted_neighbors))[0]
        min_index = self.random_state.choice(min_val)
        bin_b = neighbors[min_index]
        if bin_b < bin_a:
            i_label_keep = bin_b
            i_label_remove = bin_a
        else:
            i_label_keep = bin_a
            i_label_remove = bin_b
        return i_label_keep, i_label_remove

    def __get_most_similar_neighbor__(self,
                                      bin_a,
                                      neighbors,
                                      counted):
        mean_label = self.mean_label
        min_counted = np.min(counted[neighbors])
        if min_counted == 0 or counted[bin_a] == 0:
            i_label_keep, i_label_remove = self.__get_closest_neighbor__(
                bin_a,
                neighbors,
                counted)
        else:
            label_diff = np.absolute(mean_label[neighbors] - mean_label[bin_a])
            min_idx = np.where(label_diff == np.nanmin(label_diff))[0]
            if len(min_idx) > 0:
                neighbors = [neighbors[i] for i in min_idx]
                i_label_keep, i_label_remove = self.__get_closest_neighbor__(
                    bin_a,
                    neighbors,
                    counted)
            else:
                bin_b = neighbors[min_idx]
                if bin_b < bin_a:
                    i_label_keep = bin_b
                    i_label_remove = bin_a
                else:
                    i_label_keep = bin_a
                    i_label_remove = bin_b
        if np.isnan(mean_label[i_label_keep]) and \
                np.isnan(mean_label[i_label_remove]):
            mean_label[i_label_keep] = np.nan
        elif not np.isnan(mean_label[i_label_keep]) and \
                np.isnan(mean_label[i_label_remove]):
            pass
        elif np.isnan(mean_label[i_label_keep]) and not \
                np.isnan(mean_label[i_label_remove]):
            mean_label[i_label_keep] = mean_label[i_label_remove]
        else:
            s_k = counted[i_label_keep] * mean_label[i_label_keep]
            s_r = counted[i_label_remove] * mean_label[i_label_remove]
            s = s_r + s_k
            c = (counted[i_label_keep] + counted[i_label_remove])
            mean_label[i_label_keep] = s / c
        mask = np.ones_like(mean_label, dtype=bool)
        mask[i_label_remove] = False
        self.mean_label = self.mean_label[mask]
        return i_label_keep, i_label_remove

    def __get_closest_neighbor__(self,
                                 bin_a,
                                 neighbors,
                                 counted,
                                 unitless=True):
        if unitless:
            bin_cog = self.__calc_bin_cog_unitless__(bin_a)
            bin_cogs = [self.__calc_bin_cog_unitless__(i)
                        for i in neighbors]
        else:
            bin_cog = self.__calc_bin_cog__(bin_a)
            bin_cogs = [self.__calc_bin_cog__(i)
                        for i in neighbors]
        distance = [np.sqrt(np.sum((bin_cog - bin_i)**2))
                    for bin_i in bin_cogs]
        min_val = np.where(distance == np.min(distance))[0]
        bin_b = neighbors[self.random_state.choice(min_val)]
        if bin_b < bin_a:
            i_label_keep = bin_b
            i_label_remove = bin_a
        else:
            i_label_keep = bin_a
            i_label_remove = bin_b
        return i_label_keep, i_label_remove

    def __calc_bin_cog__(self, i_label):
        t_labels = self.i_to_t[i_label]
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        cog = np.zeros((self.n_dims, len(t_labels)))
        mean_diff = [np.mean(np.diff(self.edges[i]))
                     for i in range(self.n_dims)]
        for j, t_label in enumerate(t_labels):
            for i, bin_i in enumerate(t_label):
                try:
                    upper_edge = self.edges[i][bin_i]
                except IndexError:
                    upper_edge = None
                try:
                    lower_edge = self.edges[i][bin_i - 1]
                except IndexError:
                    lower_edge = None
                if upper_edge is None and lower_edge is None:
                    raise ValueError('Invalid label!')
                if upper_edge is None:
                    upper_edge = lower_edge + mean_diff[i]
                if lower_edge is None:
                        lower_edge = upper_edge - mean_diff[i]
                cog[i, j] = (upper_edge + lower_edge) / 2.
        return np.mean(cog, axis=1)

    def __calc_bin_cog_unitless__(self, i_label):
        t_labels = self.i_to_t[i_label]
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        cog = np.zeros((self.n_dims, len(t_labels)))
        for j, t_label in enumerate(t_labels):
            cog[:, j] = np.array(t_label)
        return np.mean(cog, axis=1)
