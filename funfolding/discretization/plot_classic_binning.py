from copy import copy

import numpy as np

import matplotlib
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_classic_binning(ax,
                              binning,
                              X=None,
                              binned=None,
                              counted=None,
                              weights=None,
                              cmap='viridis',
                              linecolor='k',
                              linewidth=2.,
                              log_c=False,
                              zorder=5):
    if binning.n_dims != 2:
        raise binning.InvalidDimension
    if X is not None:
        binned = binning.digitize(X, weights)
    binned = binned[binned > 0]
    if binned is not None:
        counted = np.bincount(binned,
                              weights=weights,
                              minlength=binning.n_bins + 1)
    cmap = matplotlib.cm.get_cmap(cmap)
    c_min = np.min(counted)
    c_max = np.max(counted)
    if log_c:
        norm = colors.LogNorm(vmin=c_min, vmax=c_max)
    else:
        norm = colors.Normalize(vmin=c_min, vmax=c_max)
    colz = cmap(norm(counted))

    for t_label, i_label in binning.t_to_i.items():
        if binning.is_oor[i_label]:
            continue
        visible_edges = []
        invisible_edges = []
        if isinstance(t_label, tuple):
            t_label = [t_label]
        for t_label in t_label:
            #  Plot edges
            var_1_h = binning.edges[0][t_label[0]]
            var_1_l = binning.edges[0][t_label[0] - 1]
            var_2_h = binning.edges[1][t_label[1]]
            var_2_l = binning.edges[1][t_label[1] - 1]
            temp_edges = []
            temp_edges.append([var_1_h, var_1_h, var_2_l, var_2_h])
            temp_edges.append([var_1_l, var_1_l, var_2_l, var_2_h])
            temp_edges.append([var_1_l, var_1_h, var_2_l, var_2_l])
            temp_edges.append([var_1_l, var_1_h, var_2_h, var_2_h])
            for e in temp_edges:
                if (e not in visible_edges):
                    if (e not in invisible_edges):
                        visible_edges.append(copy(e))
                else:
                    visible_edges.remove(e)
                    invisible_edges.append(copy(e))
        for e in visible_edges:
            ax.plot(e[:2], e[2:],
                    lw=linewidth,
                    color=linecolor,
                    ls='-',
                    zorder=zorder)
        if binning.is_oor[i_label]:
            continue

        ax.fill([var_1_l, var_1_l, var_1_h, var_1_h],
                [var_2_l, var_2_h, var_2_h, var_2_l],
                color=colz[i_label],
                zorder=zorder)
    #cbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.set_xlim(binning.edges[0][0], binning.edges[0][-1])
    ax.set_ylim(binning.edges[1][0], binning.edges[1][-1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    matplotlib.colorbar.ColorbarBase(cax,
                                     cmap=cmap,
                                     norm=norm)
