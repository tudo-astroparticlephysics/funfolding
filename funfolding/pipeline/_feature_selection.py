import time

import numpy as np
from sklearn.metrics import log_loss
from concurrent.futures import ProcessPoolExecutor

import funfolding as ff


def __train_eval__(idx,
                   X_train,
                   y_train,
                   X_validation,
                   y_validation,
                   binning,
                   X_merge=None,
                   merge_kw={}):
    binning.fit(X_train, y_train)
    if X_merge is not None:
        binning.merge(X_merge, **merge_kw)
    binned_g_validation = binning.digitize(X_validation)
    model = ff.model.LinearModel()
    model.initialize(digitized_obs=binned_g_validation,
                     digitized_truth=y_validation)
    singular_values = model.evaluate_condition()

    return idx, max(singular_values) / min(singular_values)


def recursive_feature_selection_condition_validation(X_train,
                                                     y_train,
                                                     X_validation,
                                                     y_validation,
                                                     binning,
                                                     backwards=False,
                                                     n_jobs=1,
                                                     n_folds=1,
                                                     X_merge=None,
                                                     merge_kw={}):
    '''Funciton that does a backware elimination/forward selection passed on
    the condition obtained.

    The condition is evaluated on a constant validation data sample.

    Parameters
    ----------
    X_train : {array-like, sparse matrix}, shape=(n_samples, n_features)
        The training input samples. Sparse matrix can be CSC, CSR, COO,
        DOK, or LIL. COO, DOK, and LIL are converted to CSR.

    y_train : array-like, shape=(n_samples)
        The target values for the training (class labels in classification,
        regression not yet supported).

    X_validation : {array-like, sparse matrix}, shape=(n_samples, n_features)
        The input samples used to calculate the condition. Sparse matrix can
        be CSC, CSR, COO, DOK, or LIL. COO, DOK, and LIL are converted to CSR.

    y_validation : array-like, shape=(n_samples)
        The target values for the condition calculation (class labels in
        classification, regression not yet supported).

    binning : object
        Instance of the binner which should be used. The binner must have a
        fit and digitie and merge function (if X_merge provided). E.g. the
        TreeBinningSklearn binning approach is supported

    backwards : boolean, optional
        Whether the recursive selection should be a forward selection or
        backwards elimination.

    n_jobs : int, optional
        Number of thread that should be used.

    X_merge : {array-like, sparse matrix}, shape=(n_samples, n_features)
        The input samples used for merging. Sparse matrix can be CSC, CSR, COO,
        DOK, or LIL. COO, DOK, and LIL are converted to CSR.

    merge_kw : dict, optional
        Dict with keywords and arguments used for the merge call.

    Returns
    -------
    order : list of ints, shape=(n_features,)
        List with the indices in the order of selection.

    final_condition : list of floats, shape=(n_features,)
        List with the condition for the the different feature sets.
    '''
    order = set()
    final_condition = []
    features = set(range(X_train.shape[1]))

    unused = features.difference(order)
    while len(unused) > 0:
        condition = np.ones(len(unused)) * np.inf
        task_params = []
        for i, feature in enumerate(unused):
            feature_set = order.union([feature])
            if backwards:
                feature_set = features.difference(order.union(feature_set))
            task_params.append((i, feature_set))
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            import time

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:

                def future_callback(future):
                    future_callback.finished += 1
                    if not future.cancelled():
                        i, condition_value = future.result()
                        condition[i] = condition_value
                    else:
                        raise RuntimeError('Subprocess crashed!')
                    future_callback.running -= 1

                future_callback.running = 0
                future_callback.finished = 0

                for i, feature_set in task_params:
                    while True:
                        if future_callback.running < n_jobs:
                            break
                        else:
                            time.sleep(1)
                    if X_merge is not None:
                        X_merge = X_merge[:, feature_set]
                    future = executor.submit(
                        idx=i,
                        X_train=X_train[:, feature_set],
                        y_train=y_train,
                        X_validation=X_validation[:, feature_set],
                        y_validation=y_validation,
                        binning=binning,
                        X_merge=X_merge,
                        merge_kw=merge_kw)
                    future.add_done_callback(future_callback)
                    future_callback.running += 1
        else:
            for i, feature_set in task_params:
                if X_merge is not None:
                    X_merge = X_merge[:, feature_set]
                condition[i] = __train_eval__(
                    idx=i,
                    X_train=X_train[:, feature_set],
                    y_train=y_train,
                    X_validation=X_validation[:, feature_set],
                    y_validation=y_validation,
                    binning=binning,
                    X_merge=X_merge,
                    merge_kw=merge_kw)
        selected_feature = np.argmin(condition)
        order.add(selected_feature)
        unused = features.difference(order)
    order = list(order)
    if backwards:
        order = order[::-1]
        final_condition = final_condition[::-1]
    return order, final_condition


def recursive_feature_selection_condition_pulls(X,
                                                y,
                                                binning,
                                                n_folds=5,
                                                n_events_A=-1,
                                                n_events_binning=0.2,
                                                backwards=False,
                                                n_jobs=1,
                                                n_tasks_per_job=1,
                                                X_merge=None,
                                                merge_kw={},
                                                random_state=None):
    '''Funciton that does a backware elimination/forward selection passed on
    the condition obtained.

    The condition is in a n_fold cross validation and the mean condition is
    used.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape=(n_samples, n_features)
        The training input samples. Sparse matrix can be CSC, CSR, COO,
        DOK, or LIL. COO, DOK, and LIL are converted to CSR.

    y : array-like, shape=(n_samples)
        The target values for the training (class labels in classification,
        regression not yet supported).

    binning : object
        Instance of the binner which should be used. The binner must have a
        fit and digitie and merge function (if X_merge provided). E.g. the
        TreeBinningSklearn binning approach is supported

    n_folds : int, optional
        Number of cross-validation steps.

    backwards : boolean, optional
        Whether the recursive selection should be a forward selection or
        backwards elimination.

    n_jobs : int, optional
        Number of thread that should be used.

    X_merge : {array-like, sparse matrix}, shape=(n_samples, n_features)
        The input samples used for merging. Sparse matrix can be CSC, CSR, COO,
        DOK, or LIL. COO, DOK, and LIL are converted to CSR.

    merge_kw : dict, optional
        Dict with keywords and arguments used for the merge call.

    random_state : None, int or numpy.random.RandomState, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Returns
    -------
    order : list of ints, shape=(n_features,)
        List with the indices in the order of selection.

    final_condition : list of floats, shape=(n_features,)
        List with the condition for the the different feature sets.
    '''
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    order = []
    final_condition_mean = np.zeros(X.shape[1])
    final_condition_std = np.zeros(X.shape[1])
    features = range(X.shape[1])
    unused = [i for i in features if i not in order]
    while len(unused) > 0:
        print('{}/{}'.format(len(order), len(features)))
        print(order)
        condition = np.ones((len(unused), n_folds)) * np.inf

        results = []

        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            import time
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:

                def future_callback(future):
                    print('Finished')
                    future_callback.finished += 1
                    if not future.cancelled():
                        results.extend(future.result())
                    else:
                        raise RuntimeError('Subprocess crashed!')
                    future_callback.running -= 1

                future_callback.running = 0
                future_callback.finished = 0

                def submit(job_i_tasks):
                    if len(job_i_tasks) == n_tasks_next_job:
                        while True:
                            if future_callback.running < n_jobs:
                                print('Submitted {} !'.format(
                                    len(job_i_tasks)))
                                future = executor.submit(
                                    __process_task_params__,
                                    task_params=job_i_tasks,
                                    binning=binning,
                                    X=X,
                                    y=y,
                                    X_merge=X_merge,
                                    merge_kw=merge_kw)
                                future.add_done_callback(future_callback)
                                future_callback.running += 1
                                return []
                            else:
                                time.sleep(1)
                    else:
                        return job_i_tasks

                job_i_tasks = []
                n_tasks = n_folds * len(unused)
                n_tasks_next_job = min(int(n_tasks // n_jobs), n_tasks_per_job)
                for i, feature in enumerate(unused):
                    feature_set = order + [feature]
                    if backwards:
                        feature_set = list(unused)
                        feature_set.remove(feature)
                    pull_mode_iterator = ff.pipeline.split_test_unfolding(
                        n_iterations=n_folds,
                        n_events_total=len(X),
                        n_events_test=0,
                        n_events_A=n_events_A,
                        n_events_binning=n_events_binning,
                        random_state=random_state)
                    for j, [_, A, train] in enumerate(pull_mode_iterator):
                        job_i_tasks.append(((i, j), train, A, feature_set))
                        job_i_tasks = submit(job_i_tasks)
                if len(job_i_tasks) > 0:
                    print('Submitting last tasks')
                    submit(job_i_tasks)
        else:
            for i, feature in enumerate(unused):
                feature_set = order + [feature]
                if backwards:
                    feature_set = list(unused)
                    feature_set.remove(feature)
                pull_mode_iterator = ff.pipeline.split_test_unfolding(
                    n_iterations=n_folds,
                    n_events_total=len(X),
                    n_events_test=0,
                    n_events_A=n_events_A,
                    n_events_binning=n_events_binning,
                    random_state=random_state)
                for j, [_, A, train] in enumerate(pull_mode_iterator):
                    task_params = [((i, j), train, A, feature_set)]
                    results.extend(__process_task_params__(
                        task_params=task_params,
                        binning=binning,
                        X=X,
                        y=y,
                        X_merge=X_merge,
                        merge_kw=merge_kw))
        for idx, condition_i in results:
            condition[idx[0], idx[1]] = condition_i
        mean_condition = np.mean(condition, axis=1)
        std_condition = np.std(condition, axis=1)
        idx_best = np.argmin(mean_condition)
        selected_feature = unused[idx_best]
        final_condition_mean[len(order)] = mean_condition[idx_best]
        final_condition_std[len(order)] = std_condition[idx_best]
        order.append(selected_feature)
        unused.remove(selected_feature)
    if backwards:
        order = order[::-1]
        final_condition_mean = final_condition_mean[::-1]
        final_condition_std = final_condition_std[::-1]
    return order, final_condition_mean, final_condition_std


def __process_task_params__(task_params,
                            binning,
                            X,
                            y,
                            X_merge=None,
                            merge_kw={}):
    results = []
    for idx, train, test, feature_set in task_params:
        X_train = X[train, :][:, feature_set]
        y_train = y[train]
        X_test = X[test, :][:, feature_set]
        y_test = y[test]
        if X_merge is not None:
            X_merge = X_merge[:, feature_set]
        results.append(__train_eval__(idx=idx,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_validation=X_test,
                                      y_validation=y_test,
                                      binning=binning,
                                      merge_kw=merge_kw))
    return results


def recursive_feature_selection_condition_pulls_map(X,
                                                    y,
                                                    binning,
                                                    n_folds=5,
                                                    n_events_A=-1,
                                                    n_events_binning=0.2,
                                                    backwards=False,
                                                    n_jobs=1,
                                                    n_tasks_per_job=1,
                                                    X_merge=None,
                                                    binning_y=None,
                                                    criteria='log_loss',
                                                    merge_kw={},
                                                    return_full=False,
                                                    preselection=[],
                                                    k_features=-1,
                                                    random_state=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    order = []
    order.extend(preselection)
    final_criteria_mean = np.zeros(X.shape[1])
    final_criteria_std = np.zeros(X.shape[1])
    features = range(X.shape[1])
    unused = [i for i in features if i not in order]

    work_on_task.X = X
    work_on_task.y = y
    work_on_task.X_merge = X_merge
    work_on_task.merge_kw = merge_kw
    work_on_task.binning_y = binning_y
    work_on_task.criteria = criteria

    full_criteria = []

    while len(unused) > 0 and len(order) < k_features:
        criteria = np.ones((len(unused), n_folds)) * np.inf
        feature_sets = []
        idx = []
        for i, feature in enumerate(unused):
            feature_set = (order + [feature]) * n_folds
            for j in range(n_folds):
                idx.append((i, j))
                feature_sets.append(feature_set)
        pull_mode_iterator = ff.pipeline.split_test_unfolding(
            n_iterations=len(feature_sets),
            n_events_total=len(X),
            n_events_test=0,
            n_events_A=n_events_A,
            n_events_binning=n_events_binning,
            random_state=random_state)

        if n_jobs < 2:
            for i, indices in enumerate(pull_mode_iterator):
                task_params_i = (idx[i],
                                 feature_sets[i],
                                 indices,
                                 binning.copy())
                idx, criteria_i = work_on_task(task_params=task_params_i)
                criteria[idx[0], idx[1]] = criteria_i
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                def future_callback(future):
                    future_callback.finished += 1
                    if not future.cancelled():
                        idx, criteria_i = future.result()
                        criteria[idx[0], idx[1]] = criteria_i
                    else:
                        raise RuntimeError('Subprocess crashed!')
                    future_callback.running -= 1

                future_callback.running = 0
                future_callback.finished = 0

                for i, indices in enumerate(pull_mode_iterator):
                    task_params_i = (idx[i],
                                     feature_sets[i],
                                     indices,
                                     binning.copy())
                    while True:
                        if future_callback.running < n_jobs:
                            break
                        else:
                            time.sleep(1)
                    future = executor.submit(
                        work_on_task,
                        task_params=task_params_i)
                    future.add_done_callback(future_callback)
                    future_callback.running += 1
        full_criteria.append(criteria)
        mean_criteria = np.mean(criteria, axis=1)
        std_criteria = np.std(criteria, axis=1)
        idx_best = np.argmin(mean_criteria)
        selected_feature = unused[idx_best]
        final_criteria_mean[len(order)] = mean_criteria[idx_best]
        final_criteria_std[len(order)] = std_criteria[idx_best]
        order.append(selected_feature)
        unused.remove(selected_feature)
    if backwards:
        order = order[::-1]
        final_criteria_mean = final_criteria_mean[::-1]
        final_criteria_std = final_criteria_std[::-1]

    work_on_task.X = None
    work_on_task.y = None
    work_on_task.X_merge = None
    work_on_task.merge_kw = None
    work_on_task.binning_y = None
    work_on_task.criteria = criteria
    if return_full:
        return order, final_criteria_mean, final_criteria_std, full_criteria
    else:
        return order, final_criteria_mean, final_criteria_std


def work_on_task(task_params):
    idx = task_params[0]
    feature_set = task_params[1]
    _, A_idx, binning_idx = task_params[2]
    binning_i = task_params[3]
    X_train = work_on_task.X[binning_idx, :][:, feature_set]
    y_train = work_on_task.y[binning_idx]
    X_test = work_on_task.X[A_idx, :][:, feature_set]
    binning_i.fit(X_train, y_train)
    if work_on_task.criteria.lower == 'log_loss':
        if work_on_task.binning_y is not None:
            raise ValueError('Regression not valid for criteria')
        else:
            y_test = work_on_task.y[A_idx]
        predicted_probas = binning_i.predict_proba(X_test)
        loss = log_loss(y_test, predicted_probas)
        return idx, loss
    elif work_on_task.criteria.lower == 'feature_importances':
        importance = binning_i.tree.feature_importances_[-1]
        return idx, importance
    elif work_on_task.criteria.lower == 'condition':
        if work_on_task.binning_y is not None:
            y_test = np.digitize(work_on_task.y[A_idx],
                                 bins=work_on_task.binning_y)
        else:
            y_test = work_on_task.y[A_idx]

        if work_on_task.X_merge is not None:
            X_merge_task = work_on_task.X_merge[:, feature_set]
            binning_i.merge(X_merge_task, **work_on_task.merge_kw_)
        binned_g_validation = binning_i.digitize(X_test)
        model = ff.model.LinearModel()
        model.initialize(digitized_obs=binned_g_validation,
                         digitized_truth=y_test)
        singular_values = model.evaluate_condition(normalize=False)
        return idx, max(singular_values) / min(singular_values)
