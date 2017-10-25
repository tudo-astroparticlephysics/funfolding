from __future__ import division, print_function

from copy import copy

import numpy as np

import matplotlib.colors as colors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_hexbins(ax,
                 data,
                 extent,
                 sample_weight=None,
                 cmap='viridis',
                 log_c=None,
                 hex_kwargs={},
                 zorder=4):
    if log_c:
        norm = colors.LogNorm()
    else:
        norm = colors.Normalize()

    if sample_weight is None:
        sample_weight = np.ones_like(data[:, 0])
    ax.hexbin(
        data[:, 0],
        data[:, 1],
        C=sample_weight,
        extent=extent,
        cmap=cmap,
        reduce_C_function=np.sum,
        norm=norm,
        zorder=zorder,
        **hex_kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    cb = matplotlib.colorbar.ColorbarBase(cax,
                                          cmap=cmap,
                                          norm=norm)
    return cb


class Cut:
    def __init__(self, threshold, feature, comparison):
        self.ignore = False
        self.threshold = threshold
        if feature == 0:
            self.borders = [[threshold, threshold],
                            ['inf', 'inf']]
            self.feature = 0
            self.comparison = comparison
            if comparison == 'R':
                self.side = 'Left'
            elif comparison == 'L':
                self.side = 'Right'
        elif feature == 1:
            self.borders = [['inf', 'inf'],
                            [threshold, threshold]]
            self.feature = 1
            self.comparison = comparison
            if comparison == 'R':
                self.side = 'Bottom'
            elif comparison == 'L':
                self.side = 'Top'
        else:
            print(feature)
            raise ValueError

    def intersect(self, cut2):
        if cut2 is None:
            return None
        else:
            if self.comparison == 'R' and cut2.comparison == 'R':
                cut2.borders[self.feature][0] = self.threshold
                self.borders[cut2.feature][0] = cut2.threshold
            elif self.comparison == 'L' and cut2.comparison == 'L':
                cut2.borders[self.feature][1] = self.threshold
                self.borders[cut2.feature][1] = cut2.threshold
            elif self.comparison == 'R' and cut2.comparison == 'L':
                cut2.borders[self.feature][1] = self.threshold
                self.borders[cut2.feature][0] = cut2.threshold
            elif self.comparison == 'L' and cut2.comparison == 'R':
                cut2.borders[self.feature][0] = self.threshold
                self.borders[cut2.feature][1] = cut2.threshold
            return cut2


class Node:
    def __init__(self):
        self.top = None
        self.left = None
        self.right = None
        self.bottom = None
        self.level = 0
        self.cut_list = []
        self.bin_limits = None
        self.index = -2

    def add_cut(self, threshold, feature, comparison):
        self.level += 1
        new_cut = Cut(threshold, feature, comparison)
        self.cut_list.append(copy(new_cut))
        if new_cut.side == 'Bottom':
            self.bottom = new_cut
            self.left = new_cut.intersect(self.left)
            self.right = new_cut.intersect(self.right)
        elif new_cut.side == 'Top':
            self.top = new_cut
            self.left = new_cut.intersect(self.left)
            self.right = new_cut.intersect(self.right)
        elif new_cut.side == 'Left':
            self.left = new_cut
            self.bottom = new_cut.intersect(self.bottom)
            self.top = new_cut.intersect(self.top)
        elif new_cut.side == 'Right':
            self.right = new_cut
            self.bottom = new_cut.intersect(self.bottom)
            self.top = new_cut.intersect(self.top)

    def __str__(self):
        s = ''
        if self.top is None:
            s += 'Top: None\n'
        else:
            s += 'Top: %d: %f\n' % (self.top.feature,
                                    self.top.threshold)
        if self.bottom is None:
            s += 'Bottom: None\n'
        else:
            s += 'Bottom: %d: %f\n' % (self.bottom.feature,
                                       self.bottom.threshold)
        if self.left is None:
            s += 'Left: None\n'
        else:
            s += 'Left: %d: %f\n' % (self.left.feature,
                                     self.left.threshold)
        if self.right is None:
            s += 'Right: None\n'
        else:
            s += 'Right: %d: %f\n' % (self.right.feature,
                                      self.right.threshold)
        return s

    def __plot_edges__(self, ax, limits, color='k', lw=2., ls='-', zorder=200):
        x1 = self.left
        x2 = self.right
        y1 = self.bottom
        y2 = self.top
        bin_limits = [x1, x2, y1, y2]
        for i, coord in enumerate(bin_limits):
            if coord is None:
                bin_limits[i] = limits[i]
            else:
                bin_limits[i] = coord.threshold
        self.bin_limits = bin_limits
        color = color
        zorder = zorder
        ax.plot([bin_limits[0], bin_limits[0]],
                [bin_limits[2], bin_limits[3]],
                lw=lw, color=color, ls=ls, zorder=zorder)
        ax.plot([bin_limits[1], bin_limits[1]],
                [bin_limits[2], bin_limits[3]],
                lw=lw, color=color, ls=ls, zorder=zorder)
        ax.plot([bin_limits[0], bin_limits[1]],
                [bin_limits[2], bin_limits[2]],
                lw=lw, color=color, ls=ls, zorder=zorder)
        ax.plot([bin_limits[0], bin_limits[1]],
                [bin_limits[3], bin_limits[3]],
                lw=lw, color=color, ls=ls, zorder=zorder)

    def __fill__(self, ax,
                 color,
                 zorder=199):
        bin_limits = self.bin_limits
        ax.fill([bin_limits[0], bin_limits[0], bin_limits[1], bin_limits[1]],
                [bin_limits[2], bin_limits[3], bin_limits[3], bin_limits[2]],
                color=color,
                zorder=zorder)


class TreeCrawlerPlotting:
    def __init__(self, model):
        self.tree = model.tree_
        self.node_indices = list(np.where(self.tree.feature == -2)[0])
        self.threshold = self.tree.threshold
        self.children_left = self.tree.children_left
        self.children_right = self.tree.children_right
        self.feature = self.tree.feature
        self.leaf_list = []
        self.leaf_idents = []

    def start_crawl(self):
        root = Node()
        self.get_cut_call_next(0, root)

    def get_cut_call_next(self, index, node):
        if index in self.node_indices:
            node.ident = index
            self.leaf_list.append(node)
            self.leaf_idents.append(index)
        else:
            threshold = self.tree.threshold[index]
            feature = self.tree.feature[index]
            l_node = copy(node)
            r_node = copy(node)
            l_node.add_cut(threshold, feature, 'L')
            r_node.add_cut(threshold, feature, 'R')
            self.get_cut_call_next(self.children_left[index], l_node)
            self.get_cut_call_next(self.children_right[index], r_node)

    def plot(self,
             ax,
             limits=None,
             data=None,
             cb_label='Number Events',
             sample_weight=None,
             cmap='viridis',
             linecolor='w',
             linewidth=2.,
             log_c=False,
             as_hexbins=False,
             hex_kwargs={},
             zorder=5):
        if limits is None:
            limits = [np.inf, -np.inf, np.inf, -np.inf]
            for i, leaf_i in enumerate(self.node_indices):
                leaf = self.leaf_list[self.leaf_idents.index(leaf_i)]
                if leaf.left is not None:
                    limits[0] = min(leaf.left.threshold, limits[0])
                if leaf.right is not None:
                    limits[1] = max(leaf.right.threshold, limits[1])
                if leaf.bottom is not None:
                    limits[2] = min(leaf.bottom.threshold, limits[2])
                if leaf.top is not None:
                    limits[3] = max(leaf.top.threshold, limits[3])
        for i, leaf_i in enumerate(self.node_indices):
            leaf = self.leaf_list[self.leaf_idents.index(leaf_i)]
            leaf.__plot_edges__(ax, limits,
                                color=linecolor,
                                lw=linewidth,
                                ls='-',
                                zorder=zorder)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        if data is not None:
            if as_hexbins:
                cb = plot_hexbins(ax,
                                  data,
                                  extent=limits,
                                  sample_weight=sample_weight,
                                  cmap=cmap,
                                  log_c=log_c,
                                  hex_kwargs=hex_kwargs,
                                  zorder=zorder - 1)
            else:
                cb = self.fill_bins(ax, data,
                                    cmap=cmap,
                                    log_c=log_c,
                                    zorder=zorder)
        if cb_label is not None:
            cb.set_label(cb_label)
        return limits

    def fill_bins(self, ax, data,
                  cmap='viridis',
                  c_min=None,
                  c_max=None,
                  log_c=False,
                  zorder=199):
        cmap = matplotlib.cm.get_cmap(cmap)
        if c_min is None:
            c_min = np.min(data)
        if c_max is None:
            c_max = np.max(data)
        if log_c:
            norm = colors.LogNorm(vmin=c_min, vmax=c_max)
        else:
            norm = colors.Normalize(vmin=c_min, vmax=c_max)
        colz = cmap(norm(data))
        for i, leaf_i in enumerate(self.node_indices):
            leaf = self.leaf_list[self.leaf_idents.index(leaf_i)]
            color = colz[i]
            leaf.__fill__(ax, color, zorder=zorder - 1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="8%", pad=0.05)
        cb = matplotlib.colorbar.ColorbarBase(cax,
                                              cmap=cmap,
                                              norm=norm)
        return cb


def plot_binning(ax,
                 binning,
                 X=None,
                 limits=None,
                 sample_weight=None,
                 cb_label='Number of Events',
                 cmap='viridis',
                 linecolor='0.5',
                 linewidth=1.,
                 log_c=False,
                 as_hexbins=True,
                 hex_kwargs={'rasterized': True},
                 zorder=5):
    if binning.n_dims != 2:
        raise binning.InvalidDimension
    if X is not None:
        if as_hexbins:
            data = X
            sample_weight = sample_weight
        else:
            data = binning.histogram(X, sample_weight=sample_weight)
            sample_weight = None
    else:
        data = None
    tree_crawler = TreeCrawlerPlotting(binning.tree)
    tree_crawler.start_crawl()
    limits = tree_crawler.plot(ax=ax,
                               limits=limits,
                               data=data,
                               cmap=cmap,
                               cb_label=cb_label,
                               linecolor=linecolor,
                               linewidth=linewidth,
                               log_c=log_c,
                               as_hexbins=as_hexbins,
                               hex_kwargs=hex_kwargs,
                               zorder=zorder)
    return limits
