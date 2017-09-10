import numpy as np

from sklearn.model_selection import StratifiedKFold

import funfolding as ff


def __train_eval__(idx,
                   X_train,
                   y_train,
                   X_validation,
                   y_validation,
                   clf,
                   merge_kw={}):
    clf.fit(X_train, y_train)
    if len(merge_kw) > 0:
        clf.merge(**merge_kw)
    binned_g_validation = clf.digitize(X_validation)
    model = ff.model.LinearModel()
    model.initialize(digitized_obs=binned_g_validation,
                     digitized_truth=y_validation)
    singular_values = model.evaluate_condition()

    return idx, min(singular_values) / max(singular_values)


def recursive_feature_selection_condition_validation(X_train,
                                                     y_train,
                                                     X_validation,
                                                     y_validation,
                                                     clf,
                                                     backwards=False,
                                                     n_jobs=1,
                                                     merge_kw={}):
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
                    futures.append(executor.submit(
                        idx=i,
                        X_train=X_train[:, feature_set],
                        y_train=y_train,
                        X_validation=X_validation[:, feature_set],
                        y_validation=y_validation,
                        clf=clf,
                        merge_kw=merge_kw))
                results = wait(futures)
            for future_i in results.done:
                run_result = future_i.result()
                condition[run_result[0]] = run_result[1]
        else:
            for i, feature_set in job_params:
                condition[i] = __train_eval__(
                    idx=i,
                    X_train=X_train[:, feature_set],
                    y_train=y_train,
                    X_validation=X_validation[:, feature_set],
                    y_validation=y_validation,
                    clf=clf,
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
                                             n_folds,
                                             clf,
                                             backwards=False,
                                             n_jobs=1,
                                             merge_kw={},
                                             random_state=None):
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
                for idx, train, test, feature_set in job_params:
                    futures.append(executor.submit(
                        idx=idx,
                        X_train=X[train, feature_set],
                        y_train=y[train, feature_set],
                        X_validation=X[train, feature_set],
                        y_validation=y[train, feature_set],
                        clf=clf,
                        merge_kw=merge_kw))
                results = wait(futures)
            for future_i in results.done:
                run_result = future_i.result()
                condition[run_result[0][0], run_result[0][1]] = run_result[1]
        else:
            for idx, feature_set in job_params:
                condition[idx[0], idx[1]] = __train_eval__(
                    idx=idx,
                    X_train=X[train, feature_set],
                    y_train=y[train, feature_set],
                    X_validation=X[train, feature_set],
                    y_validation=y[train, feature_set],
                    clf=clf,
                    merge_kw=merge_kw)
        selected_feature = np.argmin(np.mean(condition, axis=1))
        order.add(selected_feature)
        unused = features.difference(order)
    order = list(order)
    if backwards:
        order = order[::-1]
        final_condition = final_condition[::-1]
    return order, final_condition
