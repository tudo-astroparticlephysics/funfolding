import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class TreeBinningSklearn(object):
    def __init__(self,
                 regression=False,
                 max_features=None,
                 min_samples_split=2,
                 max_depth=None,
                 min_samples_leaf=1,
                 max_leaf_nodes=None,
                 random_state=None):

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        if regression:
            self.tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                random_state=random_state)
        else:
            self.tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                random_state=random_state)

    def fit(self,
            X,
            y,
            sample_weight=None):
        self.tree.fit(X=X,
                      y=y,
                      sample_weight=sample_weight)

    def predict(self, X):
        return self.tree.predict(X)

    def digitize(self, X):
        return self.tree.apply(X)

    def decision_path(self, X, column_names=None):
        indicator = self.tree.decision_path(X)
        return indicator
