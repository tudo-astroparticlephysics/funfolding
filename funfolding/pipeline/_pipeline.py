import numpy as np

import logging


def split_test_unfolding(n_iterations,
                         n_events_total,
                         n_events_test,
                         n_events_A=-1,
                         n_events_binning='n_events_test',
                         sample_weight=None,
                         random_state=None):
    '''Function to do split the data for multiple unfoldings of test data.


    Parameters
    ----------
    n_iterations : int
        Number of iterations

    n_iterations : int
        Number of iterations

    n_iterations : int
        Number of iterations

    n_iterations : int
        Number of iterations

    n_iterations : int
        Number of iterations
    '''
    logger = logging.getLogger('funfolding.pipeline.split_mc_test_unfolding')

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if isinstance(n_events_test, float):
        if n_events_test > 0 and n_events_test < 1:
            n_events_test = n_events_total * n_events_test
        else:
            n_events_test = int(n_events_test)
    elif not isinstance(n_events_test, int):
        raise ValueError("'n_events_test' must be either None, int or float")
    if sample_weight is None:
        logger.info("{} expected events for 'test'")
    else:
        logger.info("{} expected events for 'test' (unsampled)")

    if n_events_A is None or n_events_A == -1:
        n_events_A = None
    elif isinstance(n_events_A, float):
        n_events_A = int(n_events_total * n_events_A)
    elif not isinstance(n_events_A, int):
        raise ValueError("'n_events_A' must be either None, int or float")

    if n_events_binning is None:
        n_events_binning = 0
    elif isinstance(n_events_binning, float):
        n_events_binning = int(n_events_total * n_events_binning)
    elif isinstance(n_events_binning, str):
        if n_events_binning.lower() == 'n_events_test':
            n_events_binning = int(n_events_test)
        else:
            raise ValueError(
                "'{}'' unknown option for 'n_events_binning'".format(
                    n_events_binning))
    else:
        raise ValueError("'n_events_binning' must be either None, int or "
                         "float")

    if (n_events_test + n_events_binning + n_events_A) > n_events_total:
        raise ValueError("'n_events_test' + 'n_events_binning' + 'n_events_A' "
                         "has to be smaller than n_events_total")
    n_events_test_pulls = random_state.poisson(n_events_test,
                                            size=n_iterations)
    idx = np.arange(n_events_total)

    if sample_weight is not None:
        max_w = np.max(sample_weight)
    else:
        max_w = None

    for n_events_test_i in n_events_test_pulls:
        random_state.shuffle(idx)
        test_idx = np.sort(idx[:n_events_test_i])
        train_idx = idx[n_events_test_i:]
        if sample_weight is not None:
            selected = random_state.uniform(low=0.,
                                            high=max_w,
                                            size=n_events_test_i)
            test_idx = test_idx[selected < sample_weight]
        if n_events_binning == 0:
            if n_events_A == -1:
                A_idx = np.sort(train_idx)
            else:
                A_slice = slice(None, n_events_A)
                A_idx = np.sort(idx[A_slice])
            yield test_idx, A_idx
        else:
            binning_slice = slice(None, n_events_binning)
            binning_idx = np.sort(idx[binning_slice])
            if n_events_A == -1:
                A_slice = slice(n_events_binning, None)
            else:
                A_slice = slice(n_events_binning,
                                n_events_binning + n_events_A)
            A_idx = np.sort(idx[A_slice])
            yield test_idx, A_idx, binning_idx
