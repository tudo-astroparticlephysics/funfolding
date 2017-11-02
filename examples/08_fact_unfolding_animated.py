import logging, os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation

from funfolding import binning, model, solution

import corner


if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

    random_seed = 1337

    n_walkers = 100
    n_steps_used = 2000
    n_samples_test = 5000
    min_samples_leaf = 20
    binning_E = np.linspace(2.4, 4.2, 10)

    logging.info('Running FACT Unfolding Example Animated Gradient Descent')
    logging.info('========================================================')
    logging.info('Loading Data')

    random_state = np.random.RandomState(random_seed)

    if not os.path.isfile('fact_simulations.hdf'):
        from get_fact_simulations import download
        logging.info('Downloading FACT simulations!')
        download()
    df = pd.read_hdf('fact_simulations.hdf', 'gamma_simulation')

    idx = np.arange(len(df))
    random_state.shuffle(idx)

    test_slice = slice(0, n_samples_test)
    binning_slice = slice(n_samples_test, n_samples_test + 10 * n_samples_test)
    A_slice = slice(n_samples_test + 10 * n_samples_test, None)

    idx_test = idx[test_slice]
    idx_binning = idx[binning_slice]
    idx_A = idx[A_slice]

    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(df.loc[:, 'log10(energy)'], binning_E)
    binned_E_test = binned_E[idx_test]
    binned_E_binning = binned_E[idx_binning]
    binned_E_A = binned_E[idx_A]

    tree_obs = ["log10(E_RF)",
                "log10(Size)",
                "log10(ConcCore)",
                "log10(numPixelInShower)",
                "log10(Length)",
                "Width",
                "M3Trans",
                "M3Long",
                "m3l",
                "m3t",
                "Concentration_onePixel",
                "Concentration_twoPixel",
                "Leakage",
                "Leakage2",
                "concCOG",
                "numIslands",
                "phChargeShower_mean",
                "phChargeShower_variance",
                "phChargeShower_max"]

    obs_array = df.get(tree_obs).values

    obs_array_test = obs_array[idx_test]
    obs_array_binning = obs_array[idx_binning]
    obs_array_A = obs_array[idx_A]

    logging.info('Fitting Observable Binning')

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        min_samples_leaf=int(min_samples_leaf * 10),
        random_state=random_state)

    tree_binning.fit(obs_array_binning,
                     binned_E_binning)

    binned_g_A = tree_binning.digitize(obs_array_A)
    binned_g_test = tree_binning.digitize(obs_array_test)

    tree_model = model.LinearModel()
    tree_model.initialize(digitized_obs=binned_g_A,
                                  digitized_truth=binned_E_A)

    vec_g, vec_f = tree_model.generate_vectors(binned_g_test,
                                               binned_E_test)

    llh = solution.StandardLLH(tau=None,
                               C='thikonov')
    llh.initialize(vec_g=vec_g,
                    model=tree_model)
    logging.info('Starting Gradient Descent')
    sol_gd = solution.LLHSolutionGradientDescent(n_steps=500,
                                                 gamma=0.01)
    sol_gd.initialize(llh=llh, model=tree_model)
    sol_gd.set_x0_and_bounds()
    points, llh_values, gradient, hessian = sol_gd.fit()
    idx_best = np.argmax(llh_values)
    logging.info('Starting MCMC')
    sol_mcmc = solution.LLHSolutionMCMC(n_burn_steps=100,
                                        n_used_steps=1000,
                                        n_walkers=100,
                                        random_state=random_state)
    sol_mcmc.initialize(llh=llh, model=tree_model)
    sol_mcmc.set_x0_and_bounds(x0=points[idx_best])
    vec_f_est_mcmc, sigma_vec_f, sample, probs = sol_mcmc.fit()

    corner_fig = corner.corner(sample,
                               rasterized=True)


    def first_init():
        axes = corner_fig.axes
        pointer = -1
        line_dict = {}
        for i_y in range(points.shape[1]):
            for i_x in range(points.shape[1]):
                pointer += 1
                ax = axes[pointer]
                if i_x > i_y:
                    continue
                elif i_x == i_y:
                    line_dict[(i_x, i_y)] = ax.plot([], [], '-',
                                                    lw=2.,
                                                    color='b')[0]
                else:
                    line_dict[(i_x, i_y)] = ax.plot([], [], 'o-', color='b')[0]
        return line_dict


    line_dict = first_init()

    def clear():
        logging.info('\tClearing Plot')
        lines = []
        for k, line in line_dict.items():
            line.set_data([], [])
            lines.append(line)
        return lines


    def run(data_dict):
        pointer = -1
        lines = []
        if data_dict['finished']:
            return clear()
        for i_y in range(points.shape[1]):
            for i_x in range(points.shape[1]):
                pointer += 1
                if i_x > i_y:
                    continue
                else:
                    line_i_j = line_dict[(i_x, i_y)]
                    try:
                        p_x, p_y = data_dict[(i_x, i_y)]
                    except KeyError:
                        continue
                    else:
                        line_i_j.set_data(p_x, p_y)
                        lines.append(line_i_j)
        return lines


    def gen_data(step_size=10):
        used_points = points[::step_size]
        axes = corner_fig.axes

        total_steps = len(used_points)

        for n in range(len(used_points)):
            logging.info('\tStep: {}/{}'.format(n, total_steps))
            data_dict = {}
            data_dict['finished'] = False
            pointer = -1
            used_points_n = used_points[:n+1]
            for i_y in range(used_points_n.shape[1]):
                for i_x in range(used_points_n.shape[1]):
                    pointer += 1
                    if i_x > i_y:
                        continue
                    ax = axes[pointer]
                    x_lims = ax.get_xlim()
                    y_lims = ax.get_ylim()
                    if i_x == i_y:
                        x_pos = used_points_n[-1, i_x]
                        x_in_range = np.logical_and(x_lims[0] < x_pos,
                                                    x_lims[1] > x_pos)
                        if x_in_range:
                            data_dict[(i_x, i_y)] = [(x_pos, x_pos), y_lims]
                    else:
                        p_x = used_points_n[:, i_x]
                        p_y = used_points_n[:, i_y]
                        x_in_range = np.logical_and(x_lims[0] < p_x,
                                                    x_lims[1] > p_x)
                        y_in_range = np.logical_and(y_lims[0] < p_y,
                                                    y_lims[1] > p_y)
                        in_range = np.logical_and(x_in_range, y_in_range)
                        if sum(in_range) > 0:
                            p_x = used_points_n[in_range, i_x]
                            p_y = used_points_n[in_range, i_y]
                            data_dict[(i_x, i_y)] = (p_x, p_y)
            yield data_dict
        data_dict = {}
        data_dict['finished'] = False
        yield data_dict



    logging.info('Showing animation')

    ani = animation.FuncAnimation(corner_fig,
                                  run,
                                  gen_data,
                                  interval=50,
                                  repeat=True,
                                  blit=True,
                                  init_func=clear)
    plt.show()
