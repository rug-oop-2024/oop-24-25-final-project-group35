import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.svm import SVR as SklearnSVR

class MultipleLinearRegression(Model):

    def __init__(self, **kwargs):
        super().__init__(name="Multiple Linear Regression", model_type="regression", **kwargs)
        self.parameters = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):

        x_matrix = np.c_[np.ones((observations.shape[0], 1)), observations]
        self.parameters = (
            np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ ground_truth
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        x_matrix = np.c_[np.ones((observations.shape[0], 1)), observations]
        return x_matrix @ self.parameters



class Lasso(Model):
    
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(name="Lasso Regression", model_type="regression", **kwargs)
        self.model = SklearnLasso(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = {
            "coefficients": self.model.coef_,
            "intercept": self.model.intercept_
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)



class SupportVectorRegression(Model):
    def __init__(self, kernel='rbf', **kwargs):
        super().__init__(name="Support Vector Regression", model_type="regression", **kwargs)

        self.model = SklearnSVR(kernel=kernel)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the SVR model."""
        self.model.fit(X, y)
        self.parameters = {
            "kernel": self.model.kernel,
            "support_vectors_": self.model.support_,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the SVR model."""
        return self.model.predict(X)


