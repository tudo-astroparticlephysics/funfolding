import numpy as np
from matplotlib import pyplot as plt


def plot_llh_slice(llh, best_fit, selected_bin=None, n_points=30):
    fig, [ax_grad, ax_hess] = plt.subplots(2, 1, figsize=(24, 18))
    if selected_bin is None:
        selected_bin = np.argmax(best_fit)
    points = np.linspace(0.9 * best_fit[selected_bin],
                         1.1 * best_fit[selected_bin],
                         n_points)
    llh_values = np.zeros_like(points)
    gradient_values = np.zeros((n_points, len(best_fit)))
    hessian_values = np.zeros((n_points, len(best_fit)))

    fig, [ax_grad, ax_hess] = plt.subplots(2, 1, figsize=(24, 18))
    diff = np.diff(points)[0] / 1.5
    for i, p_i in enumerate(points):
        best_fit[selected_bin] = p_i
        llh_values[i] = llh.evaluate_llh(best_fit)
        gradient_values[i, :] = llh.evaluate_gradient(best_fit)
        hesse_matrix = llh.evaluate_hessian(best_fit)
        for j in range(len(best_fit)):
            hessian_values[i, j] = hesse_matrix[j, j]
    dx = np.ones_like(points) * diff
    dx[gradient_values[:, selected_bin] < 0] *= -1.
    dy = gradient_values[:, selected_bin] * diff
    dy[gradient_values[:, selected_bin] < 0] *= -1.

    ax_grad.quiver(points,
                   llh_values,
                   dx,
                   dy,
                   angles='xy', scale_units='xy', scale=1.)

    for i in range(len(best_fit)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        gradient_values_i = gradient_values[:, i]

        dx = np.ones_like(points) * diff
        dx[gradient_values_i < 0] *= -1.
        dy = hessian_values[:, i] * diff
        dy[gradient_values_i < 0] *= -1.
        ax.quiver(points,
                  gradient_values_i,
                  dy,
                  dy,
                  angles='xy',
                  scale_units='xy',
                  scale=1.)
        ax.set_xlabel('Events in Bin {}'.format(selected_bin))
        ax.set_ylabel(r'$\mathdefault{Gradient}_%d$' % selected_bin)
        fig.suptitle('Gradient ({}, {})'.format(i, i))
        fig.savefig('05_{}_hesse.png'.format(i))

    # ax_hess.axhline(0., color='0.5', linestyle='--')
    # ax_hess.quiver(points,
    #                gradient_values,
    #                dx,
    #                dy,
    #                angles='xy', scale_units='xy', scale=1.,
    #                pivot='tip')
    # ax_hess.set_xlabel('Events in Bin {}'.format(selected_bin))

    ax_grad.set_xlabel('Events in Bin {}'.format(selected_bin))
    # ax_hess.set_ylabel(r'$\mathdefault{Gradient}_%d$' % selected_bin)
    ax_grad.set_ylabel('Likelihood')
    return fig
