#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import itertools
import numpy as np

try:
    from astroML.density_estimation.histtools import histogram
    has_astroML = True
except ImportError:
    has_astroML = False

status_dict = {-1: 'No Binning',
               0: 'Initial Binning',
               1: 'Reduced (similar)',
               2: 'Reduced (closest)',
               3: 'Reduced (lowest)'}

class ClassicBinning(object):
    """docstring for Binning"""
    def __init__(self,
                 bins,
                 range=None,
                 oor_handle='single_bin'):
        if has_astroML:
            self.hist_func = histogram
        else:
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
        self.oor_handle = oor_handle
        self.status = -1

    def fit(self,
            X,
            sample_weight=None):
        for dim_i in range(self.n_dims):
            if sample_weight is None:
                w_i = None
            else:
                w_i = sample_weight[:, dim_i]
            self.edges.append(self.hist_func(a=X[:, dim_i],
                                             bins=self.bins[dim_i],
                                             range=self.range[dim_i],
                                             weights=w_i))
        self.status = 0

    def digitize(self, X, right=False):
        tup_label = np.zeros((len(X), self.n_dims), dtype=int)
        for dim_i in range(self.n_dims):
            tup_label[:, dim_i] = np.digitize(x=X[:, dim_i],
                                              bins=self.edges[dim_i],
                                              right=right)
        return self.convert_tup_label(tup_label)

    def convert_tup_label(self, tup_label):
        i_label = np.array((len(tup_label), self.n_dims), dtype=int)
        if self.t_to_i is None:
            self.t_to_i = self.create_conversion_dict()
        i_label = np.array([self.t_to_i.get(key, 0) for key in tup_label
                            if key not in self.t_to_i.key],
                           dtype=int)
        return i_label

    def create_conversion_dict(self):
        if self.oor_handle == 'single_bin':
            range_list = [np.arange(len(edges_i) - 1)
                          for edges_i in self.edges]
            indices = list(itertools.product(*range_list))
            self.t_to_i = {x: i+1 for i, x in enumerate(indices)}
        else:
            range_list = [np.arange(len(edges_i) + 1)
                          for edges_i in self.edges]
            indices = list(itertools.product(*range_list))
            self.t_to_i = {x: i for i, x in enumerate(indices)}
        self.i_to_t = {self.t_to_i[t]: t for t in self.t_to_i.keys()}

    def reduce(self,
               X,
               threshold,
               sample_weight=None,
               y=None,
               right=False,
               mode='closest'):
        if self.status == -1:
            raise RuntimeError("Run 'fit' before 'reduce'")
        n_merg_iterations = 0
        binned = self.digitize(X, right=right)
        if self.oor_handle == 'single_bin':
            binned = binned[binned > 0]
            if y is not None:
                y = y[binned > 0]
        max_i = max(self.i_to_t.keys())
        counted = np.bincount(binned,
                              minlength=max_i + 1,
                              weights=sample_weight)
        if mode == 'similar':
            self.status = 1
            if y is None:
                raise ValueError("For mode 'similar' labels are needed!")
            else:
                self.sum_label_ = np.zeros_like(counted, dtype=int)
                for i in range(max_i):
                    self.sum_label_[i] = np.sum(y[binned == i])
                self.__get_bin_for_merge__ = self.__get_most_similar_neighbor__
        elif mode == 'closest':
            self.status = 2
            self.__get_bin_for_merge__ = self.__get_lowest_neighbor__
        elif mode == 'lowest':
            self.status = 3
            self.__get_bin_for_merge__ = self.__get_closest_neighbor__
        else:
            raise ValueError("'closest', 'lowest' and 'similar' are "
                             "valid options for keyword 'mode'")
        while True:
            min_val = np.min(counted)
            if (min_val >= threshold) or (len(self.i_to_t.keys()) == 1):
                break
            else:
                indices_min = np.where(counted == min_val)[0]
                min_index_i = np.random.choice(indices_min)
                neighbors = self.__get_neighbors__(min_index_i)
                partner_bin = self.__get_bin_for_merge__(min_index_i,
                                                         neighbors,
                                                         counted)
                self.__merge_bins__(min_index_i, partner_bin)
                counted[partner_bin] += counted[min_index_i]
                mask = np.ones_like(counted, dtype=bool)
                mask[min_index_i] = False
                counted = counted[mask]
                n_merg_iterations += 1
        self.n_bins = len(self.i_to_t.keys())
        return n_merg_iterations

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
        t_labels = self.int_to_tuple(i_label)
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        n_bins = len(t_labels)
        cog = np.zeros((self.n_dims, n_bins))
        for j, t_label in enumerate(t_labels):
            for i, bin_i in enumerate(t_label):
                cog[i, j] = (self.limits[i][bin_i] +
                             self.limits[i][bin_i - 1]) / 2.
        return np.mean(cog, axis=1)

    def __get_neighbors__(self, i_label):
        t_labels = self.int_to_tuple(i_label)
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
        i_labels = [self.tuple_to_int(t)
                    for t in neighbors if self.tuple_to_int(t) is not None]
        return i_labels

    def __merge_bins__(self, bin_a, bin_b):
        t_labels = self.int_to_tuple(bin_a)
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        for t_label in t_labels:
            self.t_to_i[t_label] = bin_b
        for k, v in self.t_to_i.items():
            if v > bin_a:
                self.t_to_i[k] -= 1
        self.i_to_t = {}
        for k, v in self.t_to_i.items():
            if v in self.i_to_t.keys():
                if isinstance(self.i_to_t[v], list):
                    self.i_to_t[v].append(k)
                else:
                    self.i_to_t[v] = [self.i_to_t[v], k]
            else:
                self.i_to_t[v] = k

