import numpy as np
import matplotlib

matplotlib.use('Agg')
from funfolding.pipeline import split_test_unfolding


def test_split_test_unfolding():
    random_seed = 1337

    random_state = np.random.RandomState(random_seed)
    n_samples = 100000
    sample_weights = random_state.uniform(size=n_samples)

    iterator = split_test_unfolding(n_iterations=10,
                                    n_events_total=n_samples,
                                    n_events_test=30000,
                                    n_events_A=-1,
                                    n_events_binning='n_events_test',
                                    sample_weight=None,
                                    sample_test=False,
                                    sample_A=False,
                                    sample_binning=False,
                                    global_max_weight=False,
                                    random_state=random_state)
    counter = 0
    for t, A, b in iterator:
        counter += 1
        assert t is not None
        assert A is not None
        assert b is not None
        assert n_samples - len(t) - len(b) == len(A)
    assert counter == 10

    iterator = split_test_unfolding(n_iterations=10,
                                    n_events_total=n_samples,
                                    n_events_test=30000,
                                    n_events_A=-1,
                                    n_events_binning=0,
                                    sample_weight=None,
                                    sample_test=False,
                                    sample_A=False,
                                    sample_binning=False,
                                    global_max_weight=False,
                                    random_state=random_state)
    counter = 0
    for t, A in iterator:
        counter += 1
        assert t is not None
        assert A is not None
        assert n_samples - len(t) == len(A)
    assert counter == 10



    iterator = split_test_unfolding(n_iterations=10,
                                    n_events_total=n_samples,
                                    n_events_test=30000,
                                    n_events_A=30000,
                                    n_events_binning=30000,
                                    sample_weight=sample_weights,
                                    sample_test=True,
                                    sample_A=True,
                                    sample_binning=True,
                                    global_max_weight=False,
                                    random_state=random_state)
    counter = 0
    for t, A, b in iterator:
        counter += 1
        assert t is not None
        assert A is not None
        assert b is not None
        assert len(b) < 30000
        assert len(A) < 30000
        assert len(t) < 30000
    assert counter == 10
