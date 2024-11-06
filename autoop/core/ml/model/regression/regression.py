import sys
import os
# flake8: noqa: E402
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../../../')))
import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.svm import SVR as SklearnSVR


class MultipleLinearRegression(Model):
    """Implements Multiple Linear Regression for regression tasks.

    This model fits a linear model using the normal equation.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the Multiple Linear Regression model.

        Args:
            **kwargs: Additional keyword arguments passed to the Model initializer.
        """
        super().__init__(name="Multiple Linear Regression",
                         model_type="regression", **kwargs)
        self._parameters = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fits the model to the provided data.

        Args:
            observations (np.ndarray): Feature matrix for training.
            ground_truth (np.ndarray): Target values for training.
        """
        x_matrix = np.c_[np.ones((observations.shape[0], 1)), observations]
        self._parameters = (
            np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ ground_truth
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Makes predictions using the fitted model.

        Args:
            observations (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        x_matrix = np.c_[np.ones((observations.shape[0], 1)), observations]
        return x_matrix @ self._parameters


class Lasso(Model):
    """Implements Lasso Regression using sklearn's Lasso model."""

    def __init__(self, alpha=0.1, **kwargs) -> None:
        """Initializes the Lasso Regression model.

        Args:
            alpha (float, optional): Regularization strength. Defaults to 0.1.
            **kwargs: Additional keyword arguments passed to the Model initializer.
        """
        super().__init__(name="Lasso Regression",
                         model_type="regression", **kwargs)
        self._model = SklearnLasso(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the Lasso model to the provided data.

        Args:
            X (np.ndarray): Feature matrix for training.
            y (np.ndarray): Target values for training.
        """
        self._model.fit(X, y)
        self.parameters = {
            "coefficients": self._model.coef_,
            "intercept": self._model.intercept_
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the fitted Lasso model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        return self._model.predict(X)


class SupportVectorRegression(Model):
    """Implements Support Vector Regression (SVR) using sklearn's SVR model."""

    def __init__(self, kernel='rbf', **kwargs) -> None:
        """Initializes the Support Vector Regression model.

        Args:
            kernel (str, optional): Specifies the kernel type to be used in the algorithm.
                Defaults to 'rbf'.
            **kwargs: Additional keyword arguments passed to the Model initializer.
        """
        super().__init__(name="Support Vector Regression",
                         model_type="regression", **kwargs)
        self._model = SklearnSVR(kernel=kernel)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the SVR model to the provided data.

        Args:
            X (np.ndarray): Feature matrix for training.
            y (np.ndarray): Target values for training.
        """
        self._model.fit(X, y)
        self.parameters = {
            "kernel": self._model.kernel,
            "support_vectors_": self._model.support_,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the fitted SVR model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        return self._model.predict(X)
