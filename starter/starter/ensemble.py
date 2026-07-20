"""Shared model classes for the end-of-turn detector.

Kept in a separate module that is never executed directly, so pickled
models survive being saved by train.py and loaded by predict.py. If these
classes lived inside train.py, running `python train.py` would bind their
__module__ to '__main__' — and predict.py, being a *different* __main__ at
load time, would fail to unpickle them. Importing this module from either
script always resolves to the same fixed path ('ensemble'), regardless of
which file was launched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _TreeNode:
    """One node in a small randomized classification tree."""

    probability: float
    feature: int | None = None
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class RandomizedTreeEnsemble:
    """Small, deterministic, class-balanced ExtraTrees-style ensemble.

    It is intentionally shallow to control variance on the small number of
    turns.  Every tree sees a bootstrap sample; every internal node tests a
    random subset of features and a few random split thresholds.
    """

    def __init__(
        self,
        n_trees=300,
        max_depth=4,
        min_samples_leaf=3,
        feature_fraction=0.55,
        thresholds_per_feature=4,
        random_state=0,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_fraction = feature_fraction
        self.thresholds_per_feature = thresholds_per_feature
        self.random_state = random_state
        self.trees: list[_TreeNode] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x and y must have matching two-dimensional samples")
        if len(np.unique(y)) != 2:
            raise ValueError("both hold and eot examples are required for fitting")

        self._x = x
        self._y = y
        count_negative = max(1, int(np.sum(y == 0)))
        count_positive = max(1, int(np.sum(y == 1)))
        self._sample_weight = np.where(
            y == 1,
            len(y) / (2.0 * count_positive),
            len(y) / (2.0 * count_negative),
        ).astype(np.float64)
        self._prior = float(np.sum(self._sample_weight * y) / np.sum(self._sample_weight))

        rng = np.random.default_rng(self.random_state)
        self.trees = []
        total_importance = np.zeros(x.shape[1], dtype=np.float64)
        all_indices = np.arange(len(y), dtype=np.int32)
        for _ in range(self.n_trees):
            bootstrap = rng.choice(all_indices, size=len(all_indices), replace=True)
            tree_importance = np.zeros(x.shape[1], dtype=np.float64)
            self.trees.append(self._build_node(bootstrap, 0, rng, tree_importance))
            total_importance += tree_importance

        total = float(total_importance.sum())
        self.feature_importances_ = (
            total_importance / total if total > 0.0 else total_importance
        ).astype(np.float32)
        return self

    def _leaf_probability(self, indices):
        weights = self._sample_weight[indices]
        positive_weight = float(np.sum(weights * self._y[indices]))
        total_weight = float(np.sum(weights))
        # A weak prior avoids brittle exact-zero / exact-one leaves.
        return float((positive_weight + self._prior) / (total_weight + 1.0))

    def _gini(self, indices):
        if len(indices) == 0:
            return 0.0
        weights = self._sample_weight[indices]
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return 0.0
        positive = float(np.sum(weights * self._y[indices]) / total_weight)
        return 2.0 * positive * (1.0 - positive)

    def _build_node(self, indices, depth, rng, importance):
        probability = self._leaf_probability(indices)
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return _TreeNode(probability=probability)

        parent_gini = self._gini(indices)
        if parent_gini <= 1e-12:
            return _TreeNode(probability=probability)

        feature_count = max(1, int(np.ceil(self.feature_fraction * self._x.shape[1])))
        candidate_features = rng.choice(
            self._x.shape[1], size=min(feature_count, self._x.shape[1]), replace=False
        )
        parent_weight = float(np.sum(self._sample_weight[indices]))
        best_gain = 0.0
        best_feature = None
        best_threshold = 0.0
        best_left = None
        best_right = None

        for feature in candidate_features:
            values = self._x[indices, feature]
            low = float(np.min(values))
            high = float(np.max(values))
            if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-12:
                continue
            thresholds = rng.uniform(low, high, size=self.thresholds_per_feature)
            for threshold in thresholds:
                left = indices[values <= threshold]
                right = indices[values > threshold]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                left_weight = float(np.sum(self._sample_weight[left]))
                right_weight = float(np.sum(self._sample_weight[right]))
                child_gini = (
                    (left_weight * self._gini(left) + right_weight * self._gini(right))
                    / parent_weight
                )
                gain = parent_gini - child_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left
                    best_right = right

        if best_feature is None:
            return _TreeNode(probability=probability)

        importance[best_feature] += parent_weight * best_gain
        return _TreeNode(
            probability=probability,
            feature=best_feature,
            threshold=best_threshold,
            left=self._build_node(best_left, depth + 1, rng, importance),
            right=self._build_node(best_right, depth + 1, rng, importance),
        )

    @staticmethod
    def _predict_tree(tree, row):
        node = tree
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.probability

    def predict_proba(self, x):
        if not self.trees:
            raise RuntimeError("fit must be called before predict_proba")
        x = np.asarray(x, dtype=np.float32)
        probabilities = np.zeros(len(x), dtype=np.float64)
        for tree in self.trees:
            probabilities += np.fromiter(
                (self._predict_tree(tree, row) for row in x),
                dtype=np.float64,
                count=len(x),
            )
        probabilities /= len(self.trees)
        return probabilities.astype(np.float32)