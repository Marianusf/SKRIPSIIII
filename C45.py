import numpy as np
from collections import Counter


class C45:

    def __init__(self, min_samples_leaf=1):
        self.tree = None
        self.min_samples_leaf = min_samples_leaf
        self.majority_class = None

    # =========================
    # ENTROPY
    # =========================
    def entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)

        return -np.sum([
            p * np.log2(p) for p in probs if p > 0
        ])

    # =========================
    # GAIN RATIO (C4.5)
    # =========================
    def gain_ratio(self, X_column, y):

        parent_entropy = self.entropy(y)
        values = np.unique(X_column)

        weighted_entropy = 0
        split_info = 0

        for v in values:
            subset_y = y[X_column == v]

            if len(subset_y) == 0:
                continue

            weight = len(subset_y) / len(y)

            weighted_entropy += weight * self.entropy(subset_y)

            if weight > 0:
                split_info -= weight * np.log2(weight)

        info_gain = parent_entropy - weighted_entropy

        if split_info == 0:
            return 0

        return info_gain / split_info

    # =========================
    # PILIH FITUR TERBAIK
    # =========================
    def best_feature(self, X, y):

        best_feature = None
        best_gain = -1

        for col in X.columns:
            gain = self.gain_ratio(X[col].values, y.values)

            if gain > best_gain:
                best_gain = gain
                best_feature = col

        return best_feature

    # =========================
    # BUILD TREE
    # =========================
    def build_tree(self, X, y):

        # kalau semua label sama
        if len(set(y)) == 1:
            return y.iloc[0]

        # stopping: sedikit data
        if len(y) < self.min_samples_leaf:
            return Counter(y).most_common(1)[0][0]

        # tidak ada fitur
        if len(X.columns) == 0:
            return Counter(y).most_common(1)[0][0]

        best = self.best_feature(X, y)

        # kalau tidak ada split bagus
        if best is None:
            return Counter(y).most_common(1)[0][0]

        tree = {best: {}}

        for val in X[best].unique():

            subX = X[X[best] == val]
            suby = y[X[best] == val]

            subtree = self.build_tree(
                subX.drop(columns=[best]),
                suby
            )

            tree[best][val] = subtree

        return tree

    # =========================
    # FIT
    # =========================
    def fit(self, X, y):
        self.majority_class = Counter(y).most_common(1)[0][0]
        self.tree = self.build_tree(X, y)

    # =========================
    # PREDICT SATU DATA
    # =========================
    def predict_one(self, sample, tree):

        if not isinstance(tree, dict):
            return tree

        feature = list(tree.keys())[0]
        value = sample[feature]

        if value in tree[feature]:
            return self.predict_one(sample, tree[feature][value])
        else:
            # fallback ke mayoritas
            return self.majority_class

    # =========================
    # PREDICT SEMUA DATA
    # =========================
    def predict(self, X):

        predictions = []

        for _, row in X.iterrows():
            pred = self.predict_one(row, self.tree)
            predictions.append(pred)

        return predictions