#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .discretization import Discretization

import itertools
import numpy as np
from scipy.stats import itemfreq

import copy

from astroML.density_estimation.bayesian_blocks import bayesian_blocks

class ClassicBinning(Discretization):
    name = 'ClassicalBinning'
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

    def fit(self,
            X,
            y=None,
            sample_weight=None):
        super(ClassicBinning, self).fit()
        for dim_i in range(self.n_dims):
            if sample_weight is None:
                w_i = None
            else:
                w_i = sample_weight[:, dim_i]
            if self.bins[dim_i] == 'blocks':
                self.edges.append(bayesian_blocks(X[:, dim_i]))
            else:
                self.edges.append(self.hist_func(a=X[:, dim_i],
                                                 bins=self.bins[dim_i],
                                                 range=self.range[dim_i],
                                                 weights=w_i)[1])
        self.create_conversion_dict()
        self.n_bins = len(self.t_to_i.keys())

    def digitize(self, X, sample_weights=None, right=False):
        super(ClassicBinning, self).__init__()
        tup_label = np.zeros((len(X), self.n_dims), dtype=int)
        for dim_i in range(self.n_dims):
            tup_label[:, dim_i] = np.digitize(x=X[:, dim_i],
                                              bins=self.edges[dim_i],
                                              right=right)
        return self.convert_tup_label(tup_label)

    def convert_tup_label(self, tup_label):
        if self.t_to_i is None:
            self.t_to_i = self.create_conversion_dict()

        i_label = np.array([self.t_to_i.get(tuple(key), 0)
                            for key in tup_label],
                           dtype=int)
        return i_label

    def create_conversion_dict(self):
        range_list = [np.arange(len(edges_i) + 1)
                      for edges_i in self.edges]
        indices = itertools.product(*range_list)
        self.t_to_i = {x: i for i, x in enumerate(indices)}
        is_oor = lambda tup_i: any(np.array(tup_i) == 0) or \
            any([t == len(self.edges[i]) for i, t in enumerate(tup_i)])
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
        return clone

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

    def __merge__(self,
                   X,
                   min_samples=None,
                   max_bins=None,
                   sample_weight=None,
                   y=None,
                   right=False,
                   mode='closest'):
        super(ClassicBinning, self).__init__()
        n_merg_iterations = 0
        binned = self.digitize(X, right=right)
        counted = np.bincount(binned,
                              weights=sample_weight,
                              minlength=self.n_bins)
        original_counted = np.sum(counted)

        if min_samples is None and max_bins is None:
            raise ValueError("Either 'min_samples' or 'max_bins' have "
                             "to be set!")
        elif min_samples is None:
            min_samples = max(counted)
        elif max_bins is None:
            max_bins = self.n_bins


        if mode == 'similar':
            if y is None:
                raise ValueError("For mode 'similar' labels are needed!")
            else:
                if sample_weight is None:
                    w = y
                else:
                    w = y * sample_weight
                self.sum_label = np.bincount(binned,
                                             weights=w,
                                             minlength=self.n_bins)
                self.sum_label /= counted
                self.__get_bin_for_merge__ = self.__get_most_similar_neighbor__
        elif mode == 'closest':
            self.__get_bin_for_merge__ = self.__get_lowest_neighbor__
        elif mode == 'lowest':
            self.__get_bin_for_merge__ = self.__get_closest_neighbor__
        else:
            raise ValueError("'closest', 'lowest' and 'similar' are "
                             "valid options for keyword 'mode'")
        while True:
            min_val = np.min(counted)
            try:
                assert min_val <= min_samples
                n_bins = len(self.i_to_t.keys())
                assert n_bins > 1 and n_bins <= max_bins
                min_indices = np.where(counted == min_val)[0]
                min_idx = self.random_state.choice(min_indices)
                neighbors = self.__get_neighbors__(min_idx)
                partner_bin = self.__get_bin_for_merge__(min_idx,
                                                         neighbors,
                                                         counted)
                kept_i_label, removed_i_label = self.__merge_bins__(
                    min_idx,
                    partner_bin)
                counted[kept_i_label] += counted[removed_i_label]
                mask = np.ones_like(counted, dtype=bool)
                mask[removed_i_label] = False
                counted = counted[mask]
                n_merg_iterations += 1
                if np.sum(counted) != original_counted:
                    raise RuntimeError('Events sum changed!')
            except AssertionError:
                break

        self.n_bins = len(self.i_to_t.keys())
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
                               sample_weight=sample_weight,
                               y=y,
                               right=right,
                               mode=mode,
                               inplace=True)

    def __get_lowest_neighbor__(self, bin, neighbors, counted):
        counted_neighbors = counted[neighbors]
        min_val = np.where(counted_neighbors == np.min(counted_neighbors))[0]
        min_index = np.random.choice(min_val)
        return neighbors[min_index]

    def __get_most_similar_neighbor__(self, bin_a, neighbors, counted):
        counted_neighbors = counted[neighbors]
        min_counted = np.min(counted_neighbors)
        if min_counted == 0:
            return self.__get_closest_neighbor__(bin_a, neighbors, counted)
        else:
            min_index = np.argmin(self.sum_label_[neighbors] /
                                  counted[neighbors])
            bin_b = neighbors[min_index]
            s = self.sum_label_[bin_a] + self.sum_label_[bin_b]
            N = counted[bin_a] + counted[bin_b]
            self.sum_label_[bin_b] = s / N
            mask = np.ones_like(self.sum_label_, dtype=bool)
            mask[bin_a] = False
            self.sum_label_ = self.sum_label_[mask]
            return bin_b

    def __get_closest_neighbor__(self, bin_a, neighbors, counted):
        bin_cog = self.__calc_bin_cog__(bin_a)
        bin_cogs = [self.__calc_bin_cog__(i)
                    for i in neighbors]
        distance = [np.sqrt(np.sum((bin_cog - bin_i)**2))
                    for bin_i in bin_cogs]
        min_val = np.where(distance == np.min(distance))[0]
        min_index = np.random.choice(min_val)
        return neighbors[min_index]

    def __calc_bin_cog__(self, i_label):
        t_labels = self.i_to_t[i_label]
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        n_bins = len(t_labels)
        cog = np.zeros((self.n_dims, n_bins))
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

    def __get_neighbors__(self, i_label):
        t_labels = self.i_to_t[i_label]
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        neighbors = []
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
                if (upper not in neighbors) and (upper not in t_labels):
                    neighbors.append(upper)
                if (lower not in neighbors) and (lower not in t_labels):
                    neighbors.append(lower)
        i_labels = [self.t_to_i.get(t)
                    for t in neighbors
                    if t in self.t_to_i.keys()]
        assert i_label not in i_labels
        return i_labels

    def __merge_bins__(self, i_label_a, i_label_b):
        t_labels_a = self.i_to_t[i_label_a]
        if isinstance(t_labels_a, tuple):
            t_labels_a = [t_labels_a]

        t_labels_b = self.i_to_t[i_label_b]
        if isinstance(t_labels_b, tuple):
            t_labels_b = [t_labels_b]

        if i_label_a > i_label_b:
            removed_i_label = i_label_a
            kept_i_label = i_label_b
            for t_label_a_i in t_labels_a:
                self.t_to_i[t_label_a_i] = i_label_b
        else:
            removed_i_label = i_label_b
            kept_i_label = i_label_a
            for t_label_b_i in t_labels_b:
                self.t_to_i[t_label_b_i] = i_label_a

        for t_label in self.t_to_i.keys():
            if self.t_to_i[t_label] > removed_i_label:
                self.t_to_i[t_label] -= 1
        self.i_to_t = {}
        for t_label, i_label in self.t_to_i.items():
            try:
                t_labels = self.i_to_t[i_label]
                t_labels.append(t_label)
            except KeyError:
                t_labels = [t_label]
            self.i_to_t[i_label] = t_labels
        return kept_i_label, removed_i_label
