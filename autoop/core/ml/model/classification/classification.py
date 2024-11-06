import sys
import os
import numpy as np

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../')))
# flake8: noqa: E402
from collections import Counter
from sklearn.linear_model import (
    LogisticRegression as SklearnLogisticRegression)
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from autoop.core.ml.model import Model


class KNearestNeighbors(Model):
    """Implements the K-Nearest Neighbors (KNN) algorithm for classification.
    """

    def __init__(self, k=3, **kwargs) -> None:
        """Initializes the KNN classifier with a specified number of neighbors.

        Args:
            k (int, optional): Number of nearest neighbors. Defaults to 3.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If `k` is less than or equal to 0.
        """
        super().__init__(name="KNN", model_type="classification", **kwargs)
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.observations = None
        self.ground_truth = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fits the KNN model to the provided data.

        Args:
            observations (np.ndarray): Training data features.
            ground_truth (np.ndarray): Training data labels.
        """
        self.observations = observations
        self.ground_truth = ground_truth.flatten()

    def predict(self, observations: np.ndarray) -> list:
        """Predicts the labels for the given observations.

        Args:
            observations (np.ndarray): Data to predict labels for.

        Returns:
            list: Predicted labels for each observation.
        """
        return [self._predict_single(x) for x in observations]

    def _predict_single(self, observation: np.ndarray) -> int|float|str:
        """Predicts the label for a single observation based on its nearest
        neighbors.

        Args:
            observation (np.ndarray): Single data point to predict.

        Returns:
            int/float/str: The most common label among the k-nearest neighbors.
        """
        distances = np.linalg.norm(self.observations - observation, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.ground_truth[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class LogisticRegression(Model):
    """Implements Logistic Regression using sklearn's LogisticRegression."""

    def __init__(self, **kwargs) -> None:
        """Initializes the Logistic Regression model.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name="Logistic Regression",
                         model_type="classification", **kwargs)
        self._model = SklearnLogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the Logistic Regression model to the provided data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels for the given data using
        the Logistic Regression model.

        Args:
            X (np.ndarray): Data to predict labels for.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self._model.predict(X)


class DecisionTree(Model):
    """Implements a Decision Tree classifier using sklearn's
    DecisionTreeClassifier."""

    def __init__(self, max_depth=3, **kwargs) -> None:
        """Initializes the Decision Tree classifier.

        Args:
            max_depth (int, optional): Maxi depth of the tree. Defaults to 3.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name="Decision Tree",
                         model_type="classification", **kwargs)
        self._model = SklearnDecisionTree(max_depth=max_depth)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the Decision Tree model to the provided data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels for the given data using the Decision
        Tree model.

        Args:
            X (np.ndarray): Data to predict labels for.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self._model.predict(X)
