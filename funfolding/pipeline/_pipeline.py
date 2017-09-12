#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np


def sample_distribution(weights, max_w=None, random_state=None):
    logging.debug('Running sample_distribution with max_w={}'.format(max_w))
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    if max_w is None:
        max_w = np.max(weights)
    return weights <= random_state.uniform(high=max_w, size=len(weights))


def split_test_unfolding(n_iterations,
                         n_events_total,
                         n_events_test,
                         n_events_A=-1,
                         n_events_binning='n_events_test',
                         sample_weight=None,
                         sample_test=False,
                         sample_A=False,
                         sample_binning=False,
                         global_max_weight=False,
                         random_state=None):
    '''Function to do split the data for multiple unfoldings of test data.

    Parameters
    ----------
    n_iterations : int
        Number of iterations.

    n_events_total : int
        Total number of events.

    n_events_A : int or float
        Absolute or relative number of events of the test sample. If -1 all
        remaining events will be used a the sample to create A.

    n_events_A : int or float or 'n_events_test'
        Absolute or relative number of events for the binning sample.
        If set to 'n_events_test' it is set to number of events for the test
        sample.

    sample_weight : None or array_like, optional, shape=(n_events_total,)
        None for no sample weights. If array with length 'n_events_total' the
        weights can be used to sample the different parts according to the
        weights. An event is used when uniforma(0, max_weight) < sample_weight.

    sample_test : boolean, optional
        Whether the test part should be sampled according to the provided
        sample_weights.

    sample_A : boolean, optional
        Whether the A part should be sampled according to the provided
        sample_weights.

    sample_binning : boolean, optional
        Whether the binning part should be sampled according to the provided
        sample_weights.

    global_max_weight : boolean, optional
        Whether the global max weight or the maximum weight in the different
        parts should be used for the weighted sampling.

    random_state: None, int or RandomState
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.
    '''
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if isinstance(n_events_test, float):
        if n_events_test > 0 and n_events_test < 1:
            n_events_test = n_events_total * n_events_test
        else:
            n_events_test = int(n_events_test)
    elif not isinstance(n_events_test, int):
        raise ValueError("'n_events_test' must be either None, int or float")

    if n_events_A is None:
        n_events_A = -1
    elif isinstance(n_events_A, int) or isinstance(n_events_A, float):
        if n_events_A > 0 and n_events_A < 1:
            n_events_A = int(n_events_total * n_events_A)
        elif n_events_A <= 0:
            n_events_A = -1
        else:
            n_events_A = int(n_events_A)
    elif not isinstance(n_events_A, int):
        raise ValueError("'n_events_A' must be either None, int or float")

    if n_events_binning is None:
        n_events_binning = 0
    elif isinstance(n_events_binning, int) or \
            isinstance(n_events_binning, float):
        if n_events_binning > 0 and n_events_binning < 1:
            n_events_binning = int(n_events_total * n_events_binning)
        else:
            n_events_binning = int(n_events_binning)
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

    if sample_weight is not None and global_max_weight:
        max_w = np.max(sample_weight)
    else:
        max_w = None

    for n_events_test_i in n_events_test_pulls:
        random_state.shuffle(idx)
        test_idx = np.sort(idx[:n_events_test_i])
        if sample_test and sample_weight is not None:
            selected = sample_distribution(sample_weight[test_idx],
                                           max_w=max_w,
                                           random_state=random_state)
            test_idx = test_idx[selected]
        train_idx = idx[n_events_test_i:]
        if n_events_binning > 0:
            binning_slice = slice(None, n_events_binning)
            A_slice = slice(n_events_binning, None)
            binning_idx = np.sort(train_idx[binning_slice])
            if sample_binning and sample_weight is not None:
                selected = sample_distribution(sample_weight[binning_idx],
                                               max_w=max_w,
                                               random_state=random_state)
                binning_idx = binning_idx[selected]
            train_idx = train_idx[A_slice]
        else:
            binning_idx = None
        if n_events_A == -1:
            A_idx = np.sort(train_idx)
        else:
            A_idx = np.sort(train_idx[:n_events_A])
        if sample_A and sample_weight is not None:
            selected = sample_distribution(sample_weight[A_idx],
                                           max_w=max_w,
                                           random_state=random_state)
            A_idx = A_idx[selected]
        if binning_idx is None:
            yield test_idx, A_idx
        else:
            yield test_idx, A_idx, binning_idx
