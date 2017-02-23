import random

import numpy as np
from scipy import stats

import copy


random.seed(111)


def f_entropy(p):
    # Convert values to probability
    p = np.bincount(p) / float(p.shape[0])

    ep = stats.entropy(p)
    if ep == -float('inf'):
        return 0.0
    return ep


def information_gain(y, splits):
    splits_entropy = sum([f_entropy(split) * (float(split.shape[0]) /
                          y.shape[0]) for split in splits])
    return f_entropy(y) - splits_entropy


def split(X, y, value):
    left_mask = (X < value)
    right_mask = (X >= value)
    return y[left_mask], y[right_mask]


def get_split_mask(X, column, value):
    left_mask = (X[:, column] < value)
    right_mask = (X[:, column] >= value)
    return left_mask, right_mask


def split_dataset(X, y, column, value, return_X=True):
    left_mask, right_mask = get_split_mask(X, column, value)

    left = y[left_mask]
    right = y[right_mask]

    if return_X:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left, right
    else:
        return left, right


class BaseTree(object):
    leaf_list = []
    """Recursive implementation of decision tree."""
    def __init__(self,
                 parent=None,
                 regression=False,
                 criterion=information_gain,
                 cuts_happened={}):
        self.parent = parent
        self.regression = regression
        self.criterion = criterion
        self.cuts_happened = cuts_happened

        #  Set at the Beginning of the Training
        self.n_classes = None
        self.resolution = None

        #  Results of Training
        self.left_child = None
        self.right_child = None
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.leaf_idx = None

    @property
    def is_terminal(self):
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X, column):
        X = X[:, column]
        x_unique = np.unique(X)
        split_values = (x_unique[1:] + x_unique[:-1]) / 2.

        if self.resolution is not None:
            already_cutted = np.array(self.cuts_happened.get(column, []))
            resolution_column = self.resolution[column]
            for cut_i in already_cutted:
                mask = np.absolute(split_values - cut_i) >= resolution_column
                split_values = split_values[mask]
        return split_values


    def _find_best_split(self, X, y, n_features):
        """Find best feature and value for a split. Greedy algorithm."""

        # Sample random subset of features
        subset = random.sample(list(range(0, X.shape[1])), n_features)
        max_gain, max_col, max_val = None, None, None
        for column in subset:
            split_values = self._find_splits(X, column=column)
            for value in split_values:
                splits = split(X[:, column], y, value)
                gain = self.criterion(y, splits)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain

    def train(self,
              X,
              y,
              n_classes=None,
              max_features='log',
              min_samples_split=10,
              max_depth=-1,
              minimum_gain=0.01,
              resolution=None):
        """Build a decision tree from training set.

        Parameters
        ----------

        X : array-like
            Feature dataset.
        y : array-like
            Target values.
        max_features : int or None
            The number of features to consider when looking for the best        split.
        min_samples_split : int
            The minimum number of samples required to split an internal         node.
        max_depth : int
            Maximum depth of the tree.
        minimum_gain : float, default 0.01
            Minimum gain required for splitting.
        resolution : None or array-like
            Min distant for multiple cuts in the same Feature.
        """
        if not self.regression:
            self.n_classes = n_classes
        if resolution is None:
            self.resolution = np.zeros(X.shape[1])
        else:
            self.resolution = resolution
        try:
            # Exit from recursion using assert syntax
            assert (X.shape[0] > min_samples_split)
            assert (max_depth != 0)

            if max_features is None:
                max_features = X.shape[1]
            elif max_features == 'log':
                max_features = int(np.log(X.shape[1]))

            column, value, gain = self._find_best_split(
                X, y, max_features)
            assert gain is not None
            if self.regression:
                assert (gain != 0)
            else:
                assert (gain > minimum_gain)

            self.column_index = column
            self.threshold = value
            self.impurity = gain
            if column in self.cuts_happened.keys():
                self.cuts_happened[column].append(value)
            else:
                self.cuts_happened[column] = [value]

            # Split dataset
            left_X, right_X, left_y, right_y = split_dataset(
                X, y, column, value)

            # Grow left and right child
            self.left_child = self.__class__(
                parent=self,
                regression=self.regression,
                criterion=self.criterion,
                cuts_happened=copy.copy(self.cuts_happened))
            self.left_child.train(
                X=left_X,
                y=left_y,
                n_classes=self.n_classes,
                max_features=max_features,
                min_samples_split=min_samples_split,
                max_depth=max_depth-1,
                minimum_gain=minimum_gain,
                resolution=resolution)

            self.right_child = self.__class__(
                parent=self,
                regression=self.regression,
                criterion=self.criterion,
                cuts_happened=copy.copy(self.cuts_happened))
            self.right_child.train(
                X=right_X,
                y=right_y,
                n_classes=self.n_classes,
                max_features=max_features,
                min_samples_split=min_samples_split,
                max_depth=max_depth-1,
                minimum_gain=minimum_gain,
                resolution=resolution)

        except AssertionError:
            self._calculate_leaf_value(y)

    def _calculate_leaf_value(self, y):
        """Find optimal value for leaf."""
        if self.regression:
            # Mean value for regression task
            self.outcome = np.mean(y)
        else:
            # Probability for classification task
            self.outcome = np.zeros(self.n_classes)
            freq = stats.itemfreq(y)
            for idx, val in freq:
                self.outcome[idx] = val / len(y)
        self.leaf_idx = len(self.leaf_list)
        self.leaf_list.append(self)

    def predict_row(self, row):
        """Predict single row."""
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict(self, X):
        if self.regression:
            result = np.zeros(X.shape[0])
        else:
            result = np.zeros((X.shape[0], self.n_classes))
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result

    def apply(self, X):
        result = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            result[i] = self.apply_row(X[i, :])
        return result

    def apply_row(self, row):
        """Apply single row."""
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.apply_row(row)
            else:
                return self.right_child.apply_row(row)
        return self.leaf_idx


class TreeBinning(object):
    def __init__(self,
                 regression=False,
                 criterion='information_gain',
                 max_features='log',
                 min_samples_split=10,
                 max_depth=-1,
                 minimum_gain=0.01):
        class Tree(BaseTree):
            leaf_list = []


        self.TreeClass = Tree

        if criterion.lower() == 'information_gain':
            criterion = information_gain
        else:
            raise ValueError('{} as criterion unknown!'.format(criterion))
        self.root = self.TreeClass(regression=regression,
                                   criterion=criterion)
        self.tree_ops = {'max_features': max_features,
                         'min_samples_split': min_samples_split,
                         'max_depth': max_depth,
                         'minimum_gain': minimum_gain}
        self.encoding = None

    def fit(self,
              X,
              y,
              resolution=None):
        self.encoding = None
        if self.root.regression:
            n_classes = None
        else:
            if y.dtype == float:
                raise ValueError('For classification y has to be an array '
                                 'of ints!')
            train_dist = np.bincount(y)
            if any(train_dist == 0):
                y_encoded = np.zeros_like(y, dtype=int)
                new_idx = 0
                for idx, val in enumerate(train_dist):
                    if val > 0:
                        y_encoded[np.where(y == idx)[0]] = new_idx
                        new_idx += 1
                y = y_encoded
                n_classes = new_idx
            else:
                n_classes = len(train_dist)
        if resolution is None:
            resolution = np.zeros(X.shape[1])
        elif not len(resolution) == X.shape[1]:
            return ValueError('len(\'resolution\') has be equal to the number '
                              'of features!')
        self.root.train(X=X,
                        y=y,
                        n_classes=n_classes,
                        resolution=resolution,
                        **self.tree_ops)

    def predict(self, X):
        return self.root.predict(X)

    def apply(self, X):
        return self.root.apply(X)

    def get_leaf(self, idx):
        return self.root.leaf_list[idx]

    def decision_path(self, X, column_names=None):
        if isinstance(X, int):
            node = self.get_leaf(X)
        elif not isinstance(X, BaseTree):
            idx = self.root.apply_row(X)
            node = self.get_leaf(idx)
        cuts = []
        while node is not None:
            if node.is_terminal:
                cuts.append('Leaf {}'.format(node.leaf_idx))
            else:
                col = node.column_index
                if column_names is not None:
                    col = column_names[col]
                threshold = node.threshold
                cuts.append('Cut {} > {}'.format(col, threshold))
            node = node.parent
        for i, cut in enumerate(cuts[::-1]):
            print('Depth {}: {}'.format(i, cut))
