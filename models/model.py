import numpy as np
from collections import Counter
import math
import random
import pandas as pd
from sklearn.metrics import accuracy_score

# ---------------- SVM (One-vs-All) ----------------
class SVMFromScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w, self.b = np.zeros(n_features), 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class MultiClassSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            model = SVMFromScratch(self.learning_rate, self.lambda_param, self.n_iters)
            model.fit(X, y_binary)
            self.models[cls] = model

    def predict(self, X):
        return np.array([
            max(self.models.items(), key=lambda item: item[1].decision_function([x])[0])[0]
            for x in X
        ])


# ---------------- KNN ----------------
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.train_set = []

    def fit(self, X, y):
        self.train_set = [list(x) + [y_] for x, y_ in zip(X, y)]

    def predict(self, test_instance):
        distances = [
            (instance, np.linalg.norm(np.array(test_instance) - np.array(instance[:-1])))
            for instance in self.train_set
        ]
        neighbors = sorted(distances, key=lambda x: x[1])[:self.k]
        labels = [neighbor[0][-1] for neighbor in neighbors]
        return Counter(labels).most_common(1)[0][0]

    def predict_batch(self, X_test):
        return [self.predict(x) for x in X_test]


# ---------------- Logistic Regression (One-vs-Rest) ----------------
class MultiClassLogisticRegression:
    def __init__(self, n_iter = 10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres
    
    def fit(self, X, y, batch_size=64, lr=0.001, rand_seed=4, verbose=False): 
        np.random.seed(rand_seed) 
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        self.fit_data(X, y, batch_size, lr, verbose)
        return self
 
    def fit_data(self, X, y, batch_size, lr, verbose):
        i = 0
        while (not self.n_iter or i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.thres: break
            if i % 1000 == 0 and verbose: 
                print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
            i +=1
    
    def predict(self, X):
        return self.predict_(self.add_bias(X))
    
    def predict_(self, X):
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
  
    def add_bias(self,X):
        return np.insert(X, 0, 1, axis=1)
  
    def get_randon_weights(self, row, col):
        return np.zeros(shape=(row,col))

    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    
    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)
    
    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
    
    def cross_entropy(self, y, probs):
        return -1 * np.mean(y * np.log(probs))
    
# ---------------- Decision Tree ----------------
class DecisionTreeClassifierFromScratch:
    class Node:
        def __init__(self, feature=None, threshold=None, children=None, label=None):
            self.feature = feature
            self.threshold = threshold
            self.children = children or {}
            self.label = label

    def __init__(self, is_continuous_list):
        self.is_continuous_list = is_continuous_list
        self.model = None

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        data = X.copy()
        data['label'] = y.reset_index(drop=True) if hasattr(y, 'reset_index') else y
        data = data.values.tolist()
        self.model = self.build_tree(data, list(range(len(self.is_continuous_list))), self.is_continuous_list)

    def predict(self, X):
        return [self._predict_row(self.model, row) for row in X.tolist()]

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def entropy(self, data):
        labels = [row[-1] for row in data]
        total = len(labels)
        label_counts = Counter(labels)
        return -sum((count / total) * math.log2(count / total) for count in label_counts.values())

    def split_data_discrete(self, data, index, value):
        return [row for row in data if row[index] == value]

    def split_data_continuous(self, data, index, threshold):
        left = [row for row in data if row[index] <= threshold]
        right = [row for row in data if row[index] > threshold]
        return left, right

    def info_gain(self, data, subsets):
        total_entropy = self.entropy(data)
        total_len = len(data)
        weighted_entropy = sum((len(subset) / total_len) * self.entropy(subset) for subset in subsets)
        return total_entropy - weighted_entropy

    def split_info(self, subsets, total_len):
        return -sum((len(subset) / total_len) * math.log2(len(subset) / total_len)
                    for subset in subsets if len(subset) > 0)

    def gain_ratio(self, data, index, is_continuous=False):
        best_gain_ratio = -1
        best_split = None
        best_subsets = None
        total_len = len(data)

        if is_continuous:
            values = sorted(set(row[index] for row in data))
            thresholds = [(v1 + v2) / 2 for v1, v2 in zip(values[:-1], values[1:])]
            for threshold in thresholds:
                left, right = self.split_data_continuous(data, index, threshold)
                if not left or not right:
                    continue
                gain = self.info_gain(data, [left, right])
                si = self.split_info([left, right], total_len)
                ratio = gain / si if si != 0 else 0
                if ratio > best_gain_ratio:
                    best_gain_ratio = ratio
                    best_split = threshold
                    best_subsets = [left, right]
        else:
            values = set(row[index] for row in data)
            subsets = [self.split_data_discrete(data, index, val) for val in values]
            gain = self.info_gain(data, subsets)
            si = self.split_info(subsets, total_len)
            ratio = gain / si if si != 0 else 0
            best_gain_ratio = ratio
            best_split = None
            best_subsets = subsets

        return best_gain_ratio, best_split, best_subsets

    def majority_label(self, data):
        labels = [row[-1] for row in data]
        return Counter(labels).most_common(1)[0][0]

    def build_tree(self, data, features, is_continuous_list):
        labels = [row[-1] for row in data]
        if len(set(labels)) == 1:
            return self.Node(label=labels[0])
        if not features:
            return self.Node(label=self.majority_label(data))

        best_attr = -1
        best_ratio = -1
        best_split_val = None
        best_subsets = None

        for i, feature in enumerate(features):
            ratio, split_val, subsets = self.gain_ratio(data, feature, is_continuous_list[i])
            if ratio > best_ratio:
                best_attr = feature
                best_ratio = ratio
                best_split_val = split_val
                best_subsets = subsets

        if best_ratio <= 0 or best_subsets is None:
            return self.Node(label=self.majority_label(data))

        if is_continuous_list[features.index(best_attr)]:
            left, right = best_subsets
            left_child = self.build_tree(left, features, is_continuous_list)
            right_child = self.build_tree(right, features, is_continuous_list)
            return self.Node(feature=best_attr, threshold=best_split_val, children={"<=": left_child, ">": right_child})
        else:
            node = self.Node(feature=best_attr)
            for subset in best_subsets:
                if subset:
                    value = subset[0][best_attr]
                    sub_features = [f for f in features if f != best_attr]
                    sub_continuous = [is_continuous_list[i] for i, f in enumerate(features) if f != best_attr]
                    node.children[value] = self.build_tree(subset, sub_features, sub_continuous)
            return node

    def _predict_row(self, node, row):
        while node.label is None:
            val = row[node.feature]
            if node.threshold is not None:
                node = node.children["<="] if val <= node.threshold else node.children[">"]
            else:
                node = node.children.get(val)
                if node is None:
                    return None
        return node.label

# ---------------- Random Forest ----------------
class RandomForestClassifierFromScratch:
    class Node:
        def __init__(self, feature=None, threshold=None, children=None, label=None):
            self.feature = feature
            self.threshold = threshold
            self.children = children or {}
            self.label = label

    def __init__(self, n_trees=10, max_features=None, is_continuous_list=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.is_continuous_list = is_continuous_list
        self.features = None
        self.forest = []

    def entropy(self, data):
        labels = [row[-1] for row in data]
        total = len(labels)
        label_counts = Counter(labels)
        return -sum((count / total) * math.log2(count / total) for count in label_counts.values())

    def split_data_discrete(self, data, index, value):
        return [row for row in data if row[index] == value]

    def split_data_continuous(self, data, index, threshold):
        left = [row for row in data if row[index] <= threshold]
        right = [row for row in data if row[index] > threshold]
        return left, right

    def info_gain(self, data, subsets):
        total_entropy = self.entropy(data)
        total_len = len(data)
        weighted_entropy = sum((len(subset) / total_len) * self.entropy(subset) for subset in subsets)
        return total_entropy - weighted_entropy

    def split_info(self, subsets, total_len):
        return -sum((len(subset) / total_len) * math.log2(len(subset) / total_len)
                    for subset in subsets if len(subset) > 0)

    def gain_ratio(self, data, index, is_continuous=False):
        best_gain_ratio = -1
        best_split = None
        best_subsets = None
        total_len = len(data)

        if is_continuous:
            values = sorted(set(row[index] for row in data))
            thresholds = [(v1 + v2) / 2 for v1, v2 in zip(values[:-1], values[1:])]
            for threshold in thresholds:
                left, right = self.split_data_continuous(data, index, threshold)
                if not left or not right:
                    continue
                gain = self.info_gain(data, [left, right])
                si = self.split_info([left, right], total_len)
                ratio = gain / si if si != 0 else 0
                if ratio > best_gain_ratio:
                    best_gain_ratio = ratio
                    best_split = threshold
                    best_subsets = [left, right]
        else:
            values = set(row[index] for row in data)
            subsets = [self.split_data_discrete(data, index, val) for val in values]
            gain = self.info_gain(data, subsets)
            si = self.split_info(subsets, total_len)
            ratio = gain / si if si != 0 else 0
            best_gain_ratio = ratio
            best_split = None
            best_subsets = subsets

        return best_gain_ratio, best_split, best_subsets

    def _bootstrap_sample(self, data):
        return [random.choice(data) for _ in range(len(data))]

    def _majority_label(self, data):
        labels = [row[-1] for row in data]
        return Counter(labels).most_common(1)[0][0]

    def _build_tree(self, data, features, is_continuous_list):
        labels = [row[-1] for row in data]
        if len(set(labels)) == 1:
            return self.Node(label=labels[0])
        if not features:
            return self.Node(label=self._majority_label(data))

        if self.max_features and self.max_features < len(features):
            selected = random.sample(list(zip(features, is_continuous_list)), self.max_features)
            feature_candidates, continuous_candidates = zip(*selected)
        else:
            feature_candidates, continuous_candidates = features, is_continuous_list

        best_attr = -1
        best_ratio = -1
        best_split_val = None
        best_subsets = None

        for i, feature in enumerate(feature_candidates):
            ratio, split_val, subnets = self.gain_ratio(data, feature, continuous_candidates[i])
            if ratio > best_ratio:
                best_attr = feature
                best_ratio = ratio
                best_split_val = split_val
                best_subsets = subnets

        if best_ratio <= 0 or best_subsets is None:
            return self.Node(label=self._majority_label(data))

        if is_continuous_list[features.index(best_attr)]:
            left, right = best_subsets
            left_child = self._build_tree(left, features, is_continuous_list)
            right_child = self._build_tree(right, features, is_continuous_list)
            return self.Node(feature=best_attr, threshold=best_split_val, children={"<=": left_child, ">": right_child})
        else:
            node = self.Node(feature=best_attr)
            for subset in best_subsets:
                if subset:
                    value = subset[0][best_attr]
                    sub_features = [f for f in features if f != best_attr]
                    sub_continuous = [is_continuous_list[i] for i, f in enumerate(features) if f != best_attr]
                    node.children[value] = self._build_tree(subset, sub_features, sub_continuous)
            return node

    def fit(self, X, y):
        self.features = list(range(X.shape[1]))
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[i for i in range(X.shape[1])])
        
        train_data = X.copy()
        train_data['label'] = y.values if isinstance(y, pd.Series) else y
        
        data = train_data.values.tolist()
        self.forest = []
        for _ in range(self.n_trees):
            sample = self._bootstrap_sample(data)
            tree = self._build_tree(sample, self.features, self.is_continuous_list)
            self.forest.append(tree)

    def _predict_row(self, node, row):
        while node.label is None:
            val = row[node.feature]
            if node.threshold is not None:
                node = node.children["<="] if val <= node.threshold else node.children[">"]
            else:
                node = node.children.get(val)
                if node is None:
                    return None
        return node.label

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.values.tolist()
        else:
            rows = X.tolist()
        y_pred = []
        for row in rows:
            predictions = [self._predict_row(tree, row) for tree in self.forest]
            predictions = [p for p in predictions if p is not None]
            if predictions:
                y_pred.append(Counter(predictions).most_common(1)[0][0])
            else:
                y_pred.append(None)
        return y_pred