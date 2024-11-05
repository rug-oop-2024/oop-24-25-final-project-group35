import sys
import os
import numpy as np

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../')))

from collections import Counter
from sklearn.linear_model import (
    LogisticRegression as SklearnLogisticRegression)
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from autoop.core.ml.model import Model


class KNearestNeighbors(Model):
    def __init__(self, k=3, **kwargs):
        super().__init__(name="KNN", model_type="classification", **kwargs)
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.observations = None
        self.ground_truth = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self.observations = observations
        self.ground_truth = ground_truth.flatten()

    def predict(self, observations: np.ndarray):
        return [self._predict_single(x) for x in observations]

    def _predict_single(self, observation: np.ndarray):
        distances = np.linalg.norm(self.observations - observation, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.ground_truth[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class LogisticRegression(Model):
    def __init__(self, **kwargs):
        super().__init__(name="Logistic Regression",
                         model_type="classification", **kwargs)
        self._model = SklearnLogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)

    def predict(self, X: np.ndarray):
        return self._model.predict(X)


class DecisionTree(Model):
    def __init__(self, max_depth=3, **kwargs):
        super().__init__(name="Decision Tree",
                         model_type="classification", **kwargs)
        self._model = SklearnDecisionTree(max_depth=max_depth)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)

    def predict(self, X: np.ndarray):
        return self._model.predict(X)
