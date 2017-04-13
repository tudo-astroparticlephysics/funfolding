import numpy as np
from scipy import stats
import copy
from scipy.optimize import minimize, minimize_scalar, basinhopping
from sortedcontainers import SortedList


class InfoCache:
    def __init__(self):
        self.entropy = None
        self.sum_w = None
        self.X = None
        self.y = None
        self.sample_weight = None
        self.X_data = None
        self.sample_weight_data = None

def f_entropy(y, sample_weight):
    if sample_weight is None:
        sum_w = len(y)
    else:
        sum_w = np.sum(sample_weight)
    p = np.bincount(y, sample_weight) / float(sum_w)
    ep = stats.entropy(p)
    if ep == -float('inf'):
        return 0.0
    return ep




def init_min_func(X, y, sample_weight, sum_w, entropy):
    result_dict = {}
    order = np.argsort(X)
    sl = SortedList(X)
    y_sorted = y[order]
    if sample_weight is not None:
        sample_weight_i = sample_weight[order]
    else:
        sample_weight_i = None

    def min_func(split_i):
        idx = sl.bisect(split_i)
        y_r = y_sorted[idx:]
        y_l =y_sorted[:idx]
        if sample_weight_i is not None:
            w_r = sample_weight_i[idx:]
            w_l = sample_weight_i[:idx]
            sum_w_r = np.sum(w_r)
            sum_w_l = np.sum(w_l)
        else:
            w_r = None
            w_l = None
            sum_w_r = len(y_r)
            sum_w_l = len(y_l)
        ent_r = f_entropy(y_r, w_r)
        ent_l = f_entropy(y_l, w_l)
        result_dict['sum_w_r'] = sum_w_r
        result_dict['sum_w_l'] = sum_w_l
        result_dict['ent_r'] = ent_r
        result_dict['ent_l'] = ent_l
        information_gain = ent_r * sum_w_r / sum_w
        information_gain += ent_l * sum_w_l / sum_w
        information_gain /= 2.
        information_gain -= entropy

        return -information_gain

    bounds = [sl[0], sl[-1]]
    x0 = (sl[0] + sl[-1]) / 2.
    return result_dict, min_func, bounds, x0


class Node(object):
    def __init__(self,
                 base_tree,
                 path,
                 direction,
                 X,
                 y,
                 entropy,
                 sum_w,
                 sample_weight=None,
                 X_data=None,
                 sample_weight_data=None):
        self.base_tree = base_tree
        self.path = path
        self.entropy = entropy
        self.sum_w = sum_w
        self.direction = direction
        if len(path) == 0:
            self.parent = -1
        else:
            self.parent = path[-1]
        self.depth = len(path)

        self.cache_l = InfoCache()
        self.cache_r = InfoCache()
        self.feature = -2
        self.threshold = -2
        self.information_gain = None

        self.must_be_terminal = self.optimize(X,
                                              y,
                                              sample_weight,
                                              X_data,
                                              sample_weight_data)


    def optimize(self,
                 X,
                 y,
                 sample_weight,
                 X_data,
                 sample_weight_data):
        print('OPTIMIZI')
        random_state =  self.base_tree.random_state
        full_feature_list = np.arange(self.base_tree.n_features)
        max_features = self.base_tree.max_features
        if self.base_tree.max_features != -1:
            feature_list = np.sort(random_state.choice(full_feature_list,
                                                       max_features,
                                                       replace=False))
        else:
            feature_list = full_feature_list
        if X_data is not None:
            n_samples_term = X_data.shape[0]
            if sample_weight_data is None:
                frac_weight = 1.0
            else:
                sum_w_data = self.base_tree.sum_w_data
                frac_weight = np.sum(sample_weight_data) / sum_w_data
        else:
            n_samples_term = X.shape[0]
            if sample_weight is None:
                frac_weight = 1.0
            else:
                sum_w = self.base_tree.sum_w
                frac_weight = np.sum(sample_weight) / sum_w

        min_weight_fraction_leaf = self.base_tree.min_weight_fraction_leaf
        min_samples_split = self.base_tree.min_samples_split

        if n_samples_term < min_samples_split:
            return False
        if frac_weight < min_weight_fraction_leaf:
            return False
        for feature_i in feature_list:

            result_dict, min_func, bounds, x0 = init_min_func(X[:, feature_i],
                                                              y,
                                                              sample_weight,
                                                              self.sum_w,
                                                              self.entropy)

            resolution_i = self.base_tree.resolution.get(feature_i, None)
            if resolution_i is not None:
                feature_idx = np.where(self.base_tree.feature == feature_i)[0]
                cuts_done = np.array([self.base_tree.threshold[idx]
                                      for idx in feature_idx])
            else:
                cuts_done = None
            if cuts_done is None:
                def accept_func(f_new, x_new, f_old, x_old):
                    return np.logical_and(x_new < bounds[1], x_new > bounds[0])
            else:
                def accept_func(f_new, x_new, f_old, x_old):
                    far_away = np.absolute(x_new - cuts_done) > resolution_i
                    is_in_bounds = np.logical_and(x_new < bounds[1],
                                                  x_new > bounds[0])
                    return np.logical_and(is_in_bounds, far_away)
            solution = basinhopping(func=min_func,
                                    x0=x0,
                                    accept_test=accept_func)
            print(solution)
            print(result_dict)
            exit()


            for i, split_i in enumerate(X_i):
                if cuts_done is not None:
                    if any(np.absolute(split_i - cuts_done) < resolution_i):
                        continue
                entropy_r, w_r, entropy_l, w_l = self.calc_entropies(
                    y_i,
                    sample_weight_i,
                    i)

                if X_data is not None:
                    new_split_mask_data = X_data < split_i
                    n_samples_l = np.sum(~new_split_mask_data)
                    n_samples_r = np.sum(new_split_mask_data)
                else:
                    n_samples_l = len(X_i) - i
                    n_samples_r = i
                try:
                    #  Check if the splits is valid
                    min_gain = self.base_tree.min_information_gain_split
                    assert information_gain > min_gain
                    assert n_samples_l > self.base_tree.min_samples_leaf
                    assert n_samples_r > self.base_tree.min_samples_leaf
                except AssertionError:
                    pass
                else:
                    if self.information_gain is None:
                        self.information_gain = 0.
                        self.must_be_terminal = False
                    if information_gain > self.information_gain:
                        self.feature = feature_i
                        self.threshold = split_i
                        self.information_gain = information_gain

                        self.cache_l.entropy = entropy_l
                        self.cache_r.entropy = entropy_r
                        self.cache_l.sum_w = w_l
                        self.cache_r.sum_w = w_r
        if self.feature is not None:
            split_mask = X[:, self.feature] < self.threshold
            self.cache_r.X = X[split_mask]
            self.cache_l.X = X[~split_mask]
            self.cache_r.y = y[split_mask]
            self.cache_l.y = y[~split_mask]
            if sample_weight is not None:
                self.cache_r.sample_weight = X[split_mask]
                self.cache_l.sample_weight = X[~split_mask]
            if X_data is not None:
                split_mask_data = X_data[:, self.feature] < self.threshold
                self.cache_r.X_data = X[split_mask_data]
                self.cache_l.X_data = X[~split_mask_data]
                if sample_weight_data is not None:
                    self.cache_r.sample_weight_data = X[split_mask_data]
                    self.cache_l.sample_weight_data = X[~split_mask_data]
            return True
        else:
            return False

    def calc_entropies(self, y, sample_weight, i):
        y_r = y[:i]
        y_l = y[i:]
        if sample_weight is not None:
            w_r = sample_weight[:i]
            w_l = sample_weight[i:]
            sum_w_r = np.sum(w_r)
            sum_w_l = np.sum(w_l)
        else:
            w_r = None
            w_l = None
            sum_w_r = len(y_r)
            sum_w_l = len(y_l)
        ent_r = f_entropy(y_r, w_r)
        ent_l = f_entropy(y_l, w_l)
        return ent_r, sum_w_r, ent_l, sum_w_l

    def register(self, terminal=False):
        idx = len(self.base_tree.feature)
        if self.direction.lower() == 'l':
            self.base_tree.children_left[self.parent] = idx
        if self.direction.lower() == 'r':
            self.base_tree.children_right[self.parent] = idx
        self.base_tree.children_right.append(None)
        self.base_tree.children_left.append(None)
        self.base_tree.feature.append(self.feature)
        self.base_tree.threshold.append(self.threshold)
        self.base_tree.information_gain.append(self.information_gain)
        self.base_tree.parent.append(self.parent)
        self.base_tree.depth.append(self.depth)
        if terminal or self.must_be_terminal:
            self.base_tree.children_right[-1] = -1
            self.base_tree.children_left[-1] = -1
            return None, None
        else:
            new_path = copy.copy(self.path)
            new_path.append(idx)
            if terminal:
                l_node = None
                r_node = None
            else:
                l_node = Node(
                    base_tree=self.base_tree,
                    path=new_path,
                    direction='l',
                    X=self.cache_l.X,
                    y=self.cache_l.y,
                    entropy=self.cache_l.entropy,
                    sum_w=self.cache_l.sum_w,
                    sample_weight=self.cache_l.X,
                    X_data=self.cache_l.X_data,
                    sample_weight_data=self.cache_l.sample_weight_data)
                r_node = Node(
                    base_tree=self.base_tree,
                    path=new_path,
                    direction='r',
                    X=self.cache_r.X,
                    y=self.cache_r.y,
                    entropy=self.cache_r.entropy,
                    sum_w=self.cache_r.sum_w,
                    sample_weight=self.cache_r.X,
                    X_data=self.cache_r.X_data,
                    sample_weight_data=self.cache_r.sample_weight_data)
            return l_node, r_node

    def __lt__(self, partner):
        if isinstance(partner, Node):
            if partner.information_gain is None:
                return True
            else:
                return self.information_gain < partner.information_gain
        elif isinstance(partner, float):
            return self.information_gain < partner
        else:
            try:
                partner = float(partner)
            except ValueError:
                raise TypeError('Only floats and Nodes can be compared!')
            else:
                return self.information_gain < partner


class DiscretizationTree(object):
    def __init__(self,
                 max_depth=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_information_gain_split=1e-07,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 random_state=None):
        # Random State
        if not isinstance(random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        #  Options
        if max_depth is None:
            self.max_depth = np.inf
        else:
            self.max_depth = max_depth
        self.max_features = max_features
        if max_leaf_nodes is None or max_leaf_nodes == -1:
            self.max_leaf_nodes = np.inf
        else:
            self.max_leaf_nodes = max_leaf_nodes
        self.min_information_gain_split = min_information_gain_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        #  Arrays
        self.children_left = []
        self.children_right = []
        self.feature = []
        self.information_gain = []
        self.entropy = []
        self.sum_w = []
        self.threshold = []
        self.value = []
        self.weighted_n_node_samples = []
        self.parent = []
        self.depth = []
        #  Quantities
        self.node_count = -1
        self.max_depth = -1
        self.n_classes = -1
        self.n_features = -1
        self.n_outputs = -1
        self.max_depth = -1
        # Train Options
        self.resolution = {}
        self.X = None
        self.y = None
        self.sample_weight = None
        self.X_data = None
        self.sample_weight_data = None
        self.total_weight = None

    def decision_path(self, X):
        path = []
        node_pointer = 0
        while True:
            path.append(node_pointer)
            x_i = X[self.feature[node_pointer]]
            if x_i > self.threshold[node_pointer]:
                node_pointer = self.children_left[node_pointer]
            else:
                node_pointer = self.children_right[node_pointer]
            if node_pointer == -1:
                break
        return path

    def train(self,
              X,
              y,
              sample_weight=None,
              resolution=None,
              X_data=None,
              sample_weight_data=None):
        if isinstance(resolution, dict):
            self.resolution = resolution
        self.n_features = X.shape[1]
        if isinstance(self.max_features, int):
            self.max_features = min(self.max_features, self.n_features)
        else:
            self.max_features = self.n_features
        current_depth = 0
        keep_building = True
        node_list = []
        if sample_weight is None:
            sum_w = len(y)
        else:
            sum_w = np.sum(sample_weight)
        root = Node(
            base_tree=self,
            path=[],
            direction='n',
            X=X,
            y=y,
            entropy=f_entropy(y, sample_weight),
            sum_w=sum_w,
            sample_weight=sample_weight,
            X_data=X_data,
            sample_weight_data=sample_weight_data)
        node_list.append(root)
        while keep_building:
            sorted_nodes = sorted(node_list)[::-1]
            node_list = []
            for i, optimized_node in enumerate(sorted_nodes):
                unregistered_nodes = len(sorted_nodes[i:]) + len(node_list)
                n_potential_leaves = self.max_leaf_nodes - self.n_outputs
                if n_potential_leaves <= unregistered_nodes + 2:
                    terminal = True
                elif current_depth == self.max_depth:
                    terminal = True
                else:
                    terminal = False
                r_node, l_node = optimized_node.register(terminal=terminal)
                if r_node is not None and l_node is not None:
                    node_list.append(r_node)
                    node_list.append(l_node)
                    self.node_count += 2
            current_depth += 1
            if len(node_list) == 0:
                keep_building = False

class TreeBinning(object):
    def __init__(self,
                 max_depth=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_information_gain_split=1e-07,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 random_state=None):
        self.tree = DiscretizationTree(
            max_depth=max_depth,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_information_gain_split=min_information_gain_split,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            random_state=random_state)
        self.leaf_idx_mapping = None
        self.n_bins = None

    def digitize(self, X):
        decision_pathes = self.decision_path(X)
        leafyfied = [l[-1] for l in decision_pathes]
        digitized = np.array([self.leaf_idx_mapping[val_i]
                              for val_i in leafyfied])
        return np.array(digitized)

    def decision_path(self, X):
        n_events = X.shape[0]
        decision_pathes = []
        for i in range(n_events):
            decision_pathes.append(self.tree.decision_path(X[i, :]))
        return decision_pathes

    def fit(self,
            X,
            y,
            sample_weight=None,
            resolution=None,
            X_data=None,
            sample_weight_data=None):
        self.tree.train(X,
                        y,
                        sample_weight=sample_weight,
                        resolution=resolution,
                        X_data=X_data,
                        sample_weight_data=sample_weight_data)
        self.leaf_idx_mapping = {}
        is_leaf = np.where(self.tree.feature == -2)[0]
        counter = 0
        for is_leaf_i in is_leaf:
            self.leaf_idx_mapping[is_leaf_i] = counter
            counter += 1
        self.n_bins = len(self.leaf_idx_mapping)
        return self

