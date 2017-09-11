import numpy as np

from sklearn.model_selection import StratifiedKFold

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

    return idx, min(singular_values) / max(singular_values)


def recursive_feature_selection_condition_validation(X_train,
                                                     y_train,
                                                     X_validation,
                                                     y_validation,
                                                     binning,
                                                     backwards=False,
                                                     n_jobs=1,
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
        job_params = []
        for i, feature in enumerate(unused):
            feature_set = order.union([feature])
            if backwards:
                feature_set = features.difference(order.union(feature_set))
            job_params.append((i, feature_set))
        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor, wait
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for i, feature_set in job_params:
                    if X_merge is not None:
                        X_merge = X_merge[:, feature_set]
                    futures.append(executor.submit(
                        idx=i,
                        X_train=X_train[:, feature_set],
                        y_train=y_train,
                        X_validation=X_validation[:, feature_set],
                        y_validation=y_validation,
                        binning=binning,
                        X_merge=X_merge,
                        merge_kw=merge_kw))
                results = wait(futures)
            for future_i in results.done:
                run_result = future_i.result()
                condition[run_result[0]] = run_result[1]
        else:
            for i, feature_set in job_params:
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


def recursive_feature_selection_condition_cv(X,
                                             y,
                                             binning,
                                             n_folds=5,
                                             backwards=False,
                                             n_jobs=1,
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
    order = set()
    final_condition = []
    features = set(range(X.shape[1]))
    unused = features.difference(order)
    while len(unused) > 0:
        condition = np.ones((len(unused), n_folds)) * np.inf
        job_params = []
        for i, feature in enumerate(unused):
            feature_set = order.union([feature])
            if backwards:
                feature_set = features.difference(order.union(feature_set))
            strat_kfold = StratifiedKFold(n_splits=n_folds,
                                          shuffle=True,
                                          random_state=random_state)
            cv_iterator = strat_kfold.split(X, y)
            for j, [train, test] in enumerate(cv_iterator):
                job_params.append(((i, j), train, test, feature_set))
        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor, wait
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                if X_merge is not None:
                    X_merge = X_merge[:, feature_set]
                for idx, train, test, feature_set in job_params:
                    if X_merge is not None:
                        X_merge = X_merge[:, feature_set]
                    futures.append(executor.submit(
                        idx=idx,
                        X_train=X[train, feature_set],
                        y_train=y[train, feature_set],
                        X_validation=X[test, feature_set],
                        y_validation=y[test, feature_set],
                        binning=binning,
                        merge_kw=merge_kw,
                        X_merge=X_merge))
                results = wait(futures)
            for future_i in results.done:
                run_result = future_i.result()
                condition[run_result[0][0], run_result[0][1]] = run_result[1]
        else:
            for idx, feature_set in job_params:
                if X_merge is not None:
                    X_merge = X_merge[:, feature_set]
                condition[idx[0], idx[1]] = __train_eval__(
                    idx=idx,
                    X_train=X[train, feature_set],
                    y_train=y[train, feature_set],
                    X_validation=X[train, feature_set],
                    y_validation=y[train, feature_set],
                    binning=binning,
                    merge_kw=merge_kw)
        selected_feature = np.argmin(np.mean(condition, axis=1))
        order.add(selected_feature)
        unused = features.difference(order)
    order = list(order)
    if backwards:
        order = order[::-1]
        final_condition = final_condition[::-1]
    return order, final_condition
