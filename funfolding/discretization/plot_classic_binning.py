from copy import copy

import numpy as np

import matplotlib
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_classic_binning(ax,
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
    c_max = 40
    if log_c:
        norm = colors.LogNorm(vmin=0, vmax=c_max)
    else:
        norm = colors.Normalize(vmin=0, vmax=c_max)
    colz = cmap(norm(counted))

    plotted_edges = set()
    for i_label, t_labels in binning.i_to_t.items():
        if isinstance(t_labels, tuple):
            t_labels = [t_labels]
        visible_edges = set()
        invis_edges = set()
        for t_label in t_labels:
            fill_bin = True
            try:
                var_1_h = binning.edges[0][t_label[0]]
            except IndexError:
                var_1_h = None
                fill_bin = False
            try:
                var_1_l = binning.edges[0][t_label[0] - 1]
            except IndexError:
                var_1_l = None
                fill_bin = False
            try:
                var_2_h = binning.edges[1][t_label[1]]
            except IndexError:
                var_2_h = None
                fill_bin = False
            try:
                var_2_l = binning.edges[1][t_label[1] - 1]
            except IndexError:
                var_2_l = None
                fill_bin = False
            temp_edges = [(var_1_h, var_1_h, var_2_l, var_2_h),
                          (var_1_l, var_1_l, var_2_l, var_2_h),
                          (var_1_l, var_1_h, var_2_l, var_2_l),
                          (var_1_l, var_1_h, var_2_h, var_2_h)]
            for edge in temp_edges:
                if all([e_i is not None for e_i in edge]):
                    if edge not in visible_edges:
                        if edge not in invis_edges:
                            visible_edges.add(edge)
                    else:
                        visible_edges.remove(edge)
                        invis_edges.add(edge)
            if fill_bin:
                ax.fill([var_1_l, var_1_l, var_1_h, var_1_h],
                        [var_2_l, var_2_h, var_2_h, var_2_l],
                        color=colz[i_label],
                        zorder=zorder)
        for e in visible_edges:
            if e not in plotted_edges:
                ax.plot(e[:2], e[2:],
                        lw=linewidth,
                        color=linecolor,
                        ls='-',
                        zorder=zorder)
                plotted_edges.add(e)
    #cbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.set_xlim(binning.edges[0][0], binning.edges[0][-1])
    ax.set_ylim(binning.edges[1][0], binning.edges[1][-1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    matplotlib.colorbar.ColorbarBase(cax,
                                     cmap=cmap,
                                     norm=norm)
