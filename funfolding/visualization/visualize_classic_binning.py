import numpy as np

import matplotlib
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections import Counter


def plot_binning(ax,
                 binning,
                 X,
                 sample_weight=None,
                 cmap='viridis',
                 linecolor='0.5',
                 linewidth=1.,
                 log_c=False,
                 zorder=5):
    if binning.n_dims != 2:
        raise binning.InvalidDimension
    counted = binning.histogram(X, sample_weight=sample_weight)
    cmap = matplotlib.cm.get_cmap(cmap)
    c_max = np.max(counted)
    if log_c:
        norm = colors.LogNorm(vmin=0, vmax=c_max)
    else:
        norm = colors.Normalize(vmin=0, vmax=c_max)
    colz = cmap(norm(counted))

    plotted_edges = set()
    for i_label, t_labels in binning.i_to_t.items():
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        edges = []
        for t_label in t_labels:
            fill_bin = True

            idx = t_label[0]
            if idx < 0 or idx >= len(binning.edges[0]):
                var_1_h = None
                fill_bin = False
            else:
                var_1_h = binning.edges[0][idx]

            idx = t_label[0] - 1
            if idx < 0 or idx >= len(binning.edges[0]):
                var_1_l = None
                fill_bin = False
            else:
                var_1_l = binning.edges[0][idx]

            idx = t_label[1]
            if idx < 0 or idx >= len(binning.edges[1]):
                var_2_h = None
                fill_bin = False
            else:
                var_2_h = binning.edges[1][idx]

            idx = t_label[1] - 1
            if idx < 0 or idx >= len(binning.edges[1]):
                var_2_l = None
                fill_bin = False
            else:
                var_2_l = binning.edges[1][idx]
            temp_edges = [(var_1_h, var_1_h, var_2_l, var_2_h),
                          (var_1_l, var_1_l, var_2_l, var_2_h),
                          (var_1_l, var_1_h, var_2_l, var_2_l),
                          (var_1_l, var_1_h, var_2_h, var_2_h)]
            for edge in temp_edges:
                if all([e_i is not None for e_i in edge]):
                    edges.append(edge)
            if fill_bin:
                ax.fill([var_1_l, var_1_l, var_1_h, var_1_h],
                        [var_2_l, var_2_h, var_2_h, var_2_l],
                        color=colz[i_label],
                        zorder=zorder)
        edges_dict = Counter(edges)
        for e, freq in edges_dict.items():
            if freq == 1:
                if e not in plotted_edges:
                    ax.plot(e[:2], e[2:],
                            lw=linewidth,
                            color=linecolor,
                            ls='-',
                            zorder=zorder)
                plotted_edges.add(e)
    ax.set_xlim(binning.edges[0][0], binning.edges[0][-1])
    ax.set_ylim(binning.edges[1][0], binning.edges[1][-1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    matplotlib.colorbar.ColorbarBase(cax,
                                     cmap=cmap,
                                     norm=norm)

def mark_bin(ax, binning, i_label, color='r', linewidth=1., zorder=6):
    t_labels = binning.i_to_t[i_label]
    if isinstance(t_labels, tuple):
        t_labels = [t_labels]
    edges = []

    for t_label in t_labels:
        idx = t_label[0]
        if idx < 0 or idx >= len(binning.edges[0]):
            var_1_h = None
        else:
            var_1_h = binning.edges[0][idx]

        idx = t_label[0] - 1
        if idx < 0 or idx >= len(binning.edges[0]):
            var_1_l = None
        else:
            var_1_l = binning.edges[0][idx]

        idx = t_label[1]
        if idx < 0 or idx >= len(binning.edges[1]):
            var_2_h = None
        else:
            var_2_h = binning.edges[1][idx]

        idx = t_label[1] - 1
        if idx < 0 or idx >= len(binning.edges[0]):
            var_2_l = None
        else:
            var_2_l = binning.edges[1][idx]
        temp_edges = [(var_1_h, var_1_h, var_2_l, var_2_h),
                      (var_1_l, var_1_l, var_2_l, var_2_h),
                      (var_1_l, var_1_h, var_2_l, var_2_l),
                      (var_1_l, var_1_h, var_2_h, var_2_h)]
        for edge in temp_edges:
            if all([e_i is not None for e_i in edge]):
                edges.append(edge)
    edges_dict = Counter(edges)
    for e, freq in edges_dict.items():
        if freq == 1:
            ax.plot(e[:2], e[2:],
                    lw=linewidth,
                    color=color,
                    ls='-',
                    zorder=zorder)

