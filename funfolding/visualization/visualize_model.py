from __future__ import division, print_function
import numpy as np

import matplotlib.colors as colors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_A(ax,
           model,
           cmap='viridis',
           zorder=5):
    mean_guess = np.argmax(model.A,axis=1)
    order = np.argsort(mean_guess)
    ax.imshow(model.A[order],
              aspect='equal',
              norm = colors.LogNorm())
