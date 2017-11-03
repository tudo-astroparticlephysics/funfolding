import numpy as np
import copy
from distutils.version import StrictVersion

import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from ._binning import Binning
import warnings


old_sklean = StrictVersion("0.19.0") >= StrictVersion(sklearn.__version__)


def __sample_uniform__(y, sample_weight=None, random_state=None):
    """Function used to sample a uniform distribution from a binned y.

    Parameters
    ----------
    y : numpy.intarray, shape=(n_samples)
        Array of the true classification labels.

    y : numpy.floatarray, shape=(n_samples)
        Event weights.

    random_state : None, int or numpy.random.RandomState, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.


    Returns
    -------
    mask: list of bools
        A boolean mask for y. True for samples that should be kept.
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    freq = np.bincount(y, weights=sample_weight)
    mask = freq > 0
    if sample_weight is None:
        freq = freq.astype(float)
    freq /= np.min(freq[mask])
    rnd = random_state.uniform(size=len(y)) * freq[y]
    return rnd <= 1.


def __get_parents__(tree):
    parents = tree.children_left.copy()
    for i, (c_l, c_r) in zip(tree.children_left, tree.children_right):
        if c_l != -1:
            parents[c_l] = i
        if c_r != -1:
            parents[c_r] = i
    grand_parents = parents.copy()
    for i, p in enumerate(parents):
        if p != -1:
            grand_parent_idx = parents[p]
            if grand_parent_idx != -1:
                grand_parents[i] = parents
    return parents, grand_parents


def get_family(tree):
    """Function to get the relations between nodes in a sklean tree.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        Tree form which the relations should be extracted

    Returns
    -------
    parents : array of ints, shape=(n_nodes)
        List of the parents of each node. -1 indicates no parents and
        should only appear for the first node.

    grand_parents : array of ints, shape=(n_nodes)
        List of the parents of each node. -1 indicates no parents and
        should only appear for the first node.

    siblings : array of ints, shape=(n_nodes)
        List of the parents of each node. -1 indicates no parents and
        should only appear for the first node.
    """

    parents = np.ones_like(tree.children_left) * -1
    grand_parents = np.ones_like(tree.children_left) * -1
    siblings = np.ones_like(tree.children_left, dtype=int) * -1
    for i, (c_l, c_r) in enumerate(zip(tree.children_left,
                                       tree.children_right)):
        siblings[c_l] = c_r
        siblings[c_r] = c_l
        if c_l != -1:
            parents[c_l] = i
        if c_r != -1:
            parents[c_r] = i
        p = parents[i]
        if p != -1:
            grand_parent_idx = parents[p]
            grand_parents[i] = grand_parent_idx
        else:
            grand_parents[i] = -1
    return parents, grand_parents, siblings


def delete_node(tree, node_idx):
    """Function to mark an unused node in a tree.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        Tree form which the relations should be extracted

    node_idx : int
        Index of the node in the tree.
    """
    tree.children_right[node_idx] = -4
    tree.children_left[node_idx] = -4
    tree.feature[node_idx] = -4
    tree.threshold[node_idx] = -4


def remove_node(tree, node_idx):
    """Function to remove a node from the tree.
    The node is mark as unused and the parent node is replaced with the
    sibling. After removing a node the 'predict', 'predict_proba' won't
    produce reasonable results. This still has to be implemented.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        Tree form which the relations should be extracted

    node_idx : int
        Index of the node in the tree.
    """
    parents, grand_parents, siblings = get_family(tree)

    parent_idx = parents[node_idx]
    sibling_idx = siblings[node_idx]

    tree.children_right[parent_idx] = tree.children_right[sibling_idx]
    tree.children_left[parent_idx] = tree.children_left[sibling_idx]
    tree.feature[parent_idx] = tree.feature[sibling_idx]
    tree.threshold[parent_idx] = tree.threshold[sibling_idx]
    delete_node(tree, sibling_idx)
    delete_node(tree, node_idx)
    return parent_idx, sibling_idx


class TreeBinningSklearn(Binning):
    """Class to get a discrete representation of a event dataset.

    The representation is optimized with either a decision tree or a
    regression tree.

    Parameters
    ----------
    regression : bool, optional (default=False)
        ``True`` for a DecisionTreeClassifier.
        ``False`` for DecisionTreeRegressor.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    uniform : boolean, optional (default=False)
        Only valid for a classification tree. If True a uniform
        distribution is sampled before the model is trained.

    boosted : None or name of boosting algorithm, optional (default=None)
        If None no boosting is used. For boosting set to the name of
        the wanted algorithms. The base algorithm is the sklearn
        AdaBoost algorithm.
         For regression ``linear``, ``square`` and ``exponential`` can be used.
        For classification ``SAMME``, ``SAMME.R`` are valid. For further
        explanations check the AdaBoost documentation in sklearn.

    n_estimators : int, optional (default=50)
        Number of estimators when boosting is activated.
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        If boosting is not activated this option will be ignored.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    ensemble_select : [``last``,``best``], optional (default=best)
        Determines which tree should be used. ``best`` is the tree with
        the highest estimator weight and ``last`` is the tree with the
        strongest boosting.

    random_state : None, int or numpy.random.RandomState, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    tree : sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor
        Instance of the internal classifier.

    boosted : None or sklearn.ensemble.AdaBoostClassifier/AdaBoostRegressor
        None for no boosting or AdaBoost instance.

    random_state : numpy.random.RandomState
        State of the random number generator.

    n_bins : int
        Number of bins in the final discretization

    merged : boolean
        Indicates if the models was merge after the fit.
    """
    def __init__(self,
                 regression=False,
                 max_features=None,
                 min_samples_split=2,
                 max_depth=None,
                 min_samples_leaf=1,
                 max_leaf_nodes=None,
                 min_weight_fraction_leaf=0.,
                 min_impurity_decrease=0.,
                 boosted=None,
                 uniform=False,
                 n_estimators=50,
                 learning_rate=1.0,
                 ensemble_select='best',
                 random_state=None):
        super(TreeBinningSklearn, self).__init__()
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state
        self.regression = regression
        if self.regression and uniform:
            warnings.warn(
                'Uniform smapling is only supported for classifcation')
            self.uniform = False
        else:
            self.uniform = uniform

        if regression:
            if old_sklean:
                if min_impurity_decrease != 0.:
                    warnings.warn('min_impurity_decrease is supported '
                                  ' only in sklearn version >= 0.19.0')
                self.tree = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    max_features=max_features,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    random_state=random_state)
            else:
                self.tree = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    max_features=max_features,
                    min_impurity_decrease=min_impurity_decrease,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    random_state=random_state)
            if boosted in ['linear', 'square', 'exponential']:
                self.boosted = AdaBoostRegressor(
                    base_estimator=self.tree,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    algorithm=boosted,
                    random_state=random_state)

            elif boosted is not None:
                raise ValueError(
                    '\'boosted\' should be None for no boosting '
                    'or either \'linear\', \'square\', \'exponential\' for a '
                    'boosted regression.')
            else:
                self.boosted = None
        else:
            if old_sklean:
                if min_impurity_decrease != 0.:
                    warnings.warn('min_impurity_decrease is supported '
                                  ' only in sklearn version >= 0.19.0')
                self.tree = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    max_features=max_features,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    random_state=random_state)
            else:
                self.tree = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    max_features=max_features,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    min_impurity_decrease=min_impurity_decrease,
                    random_state=random_state)
            if boosted in ['SAMME', 'SAMME.R']:
                self.boosted = AdaBoostClassifier(
                    base_estimator=self.tree,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    algorithm=boosted,
                    random_state=random_state)
            elif boosted is not None:
                raise ValueError(
                    '\'boosted\' should be None for no boosting '
                    'or either \'SAMME\' or \'SAMME.R\' for a '
                    'boosted classification.')
            else:
                self.boosted = None
        self.ensemble_select_ = ensemble_select.lower()
        self.leaf_idx_mapping_ = None
        self.n_bins = None
        self.n_dims = None
        self.merged = False

    def initialize(self,
                   X,
                   y,
                   sample_weight=None,
                   uniform=False):
        """Build a (boosted) classification/regression tree from the training
        set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=(n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like, shape=(n_samples)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : None or array-like, optional shape=(n_samples)
            Sample weights. If None (default) and active boosting, the sample
            weights are initialized to 1 / n_samples.

        Returns
        -------
        self : object
            Returns self.
        """
        super(TreeBinningSklearn, self).initialize()
        self.n_dims = X.shape[1]
        if self.uniform and not self.regression:
            mask = __sample_uniform__(y,
                                      sample_weight=sample_weight,
                                      random_state=self.random_state)
            y = y[mask]
            X = X[mask]
        if self.boosted is not None:
            if self.ensemble_select_.lower() not in ['best', 'last']:
                raise ValueError(
                    '\'ensemble_select\' must be \'best\' or \'last\'!')
            self.boosted.fit(X=X,
                             y=y,
                             sample_weight=sample_weight)
            if self.ensemble_select_ == 'best':
                tree_idx = np.argmax(self.boosted.estimator_weights_)
            elif self.ensemble_select_ == 'last':
                tree_idx = -1
            self.tree = self.boosted.estimators_[tree_idx]
        else:
            self.tree.fit(X=X,
                          y=y,
                          sample_weight=sample_weight)

        self.__generate_leaf_mapping__()
        return self

    def __generate_leaf_mapping__(self):
        self.leaf_idx_mapping_ = {}
        is_leaf = np.where(self.tree.tree_.feature == -2)[0]
        counter = 0
        for is_leaf_i in is_leaf:
            self.leaf_idx_mapping_[is_leaf_i] = counter
            counter += 1
        self.n_bins = len(self.leaf_idx_mapping_)

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        if self.merged:
            warnings.warn('The model was merged after the trainng and '
                          'might give unreasonable predictions.')
        return self.tree.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        RuntimeError
            If predict proba is called for a regression model.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if self.merged:
            warnings.warn('The model was merged after the trainng and '
                          'might give unreasonable predictions.')
        if self.regression:
            raise RuntimeError('Can only be used for a classification!')
        return self.tree.predict_proba(X)

    def digitize(self, X):
        """Return a digitized version of the input sample X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        digitized : numpy.intarray, shape=(n_samples)
            Returns the bin number of each sample. The mapping between
            tree node and bin is stored in ``leaf_idx_mapping_``.
        """
        super(TreeBinningSklearn, self).digitize()
        leafyfied = self.tree.apply(X)
        digitized = np.array([self.leaf_idx_mapping_[val_i]
                              for val_i in leafyfied])
        return digitized

    def decision_path(self, X):
        """Return the decision path in the tree

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        indicator = self.tree.decision_path(X)
        return indicator

    def copy(self):
        """Return a deepcopy of the instance.

        Returns
        -------
        indicator : TreeBinningSklearn
            Copy of the instance.
        """
        clone = TreeBinningSklearn(random_state=self.random_state)
        clone.tree = copy.deepcopy(self.tree)
        clone.uniform = bool(self.uniform)
        return clone

    def merge(self, X, threshold, inplace=True):
        """Modifies the tree to ensure that in each leaf/bin are at least
        ``threshold`` samples.

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        threshold : int
            Minimal number of samples in each leaf.

        inplace : bool, optinal
            If 'True' the instance is merged and with inplace=True a
            copy of the instance is merged and returned

        Returns
        -------
        merged_binning : TreeBinningSklearn
            Instance with merged bins.
        """
        if inplace:
            return self.__merge__(X=X, threshold=threshold)
        else:
            clone = self.copy()
            return clone.__merge__(X=X, threshold=threshold)

    def __merge__(self, X, threshold):
        super(TreeBinningSklearn, self).merge()
        tree = self.tree.tree_

        def find_withered_leaf():
            leafyfied = self.tree.apply(X)
            occureance = np.bincount(leafyfied, minlength=tree.node_count)
            is_leaf = tree.children_right == -1
            is_below = occureance < threshold
            is_leaf_below = np.logical_and(is_leaf, is_below)
            if any(is_leaf_below):
                is_leaf_below_idx = np.where(is_leaf_below)[-1]
                idx_min_leaf = np.argmin(occureance[is_leaf_below_idx[::-1]])
                return is_leaf_below_idx[idx_min_leaf]
            else:
                return None
        while True:
            idx = find_withered_leaf()
            if idx is None:
                break
            else:
                remove_node(tree, idx)
                self.__generate_leaf_mapping__()
        return self
