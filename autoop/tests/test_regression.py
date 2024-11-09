import unittest
import numpy as np
import sys
import os
# flake8: noqa: E402
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../')))
from autoop.core.ml.model.regression.regression import (
    MultipleLinearRegression, Lasso, SupportVectorRegression)


class TestRegression(unittest.TestCase):

    def setUp(self) -> None:
        """Example data for testing."""
        self.X_train = np.array([[1], [2], [3], [4]])
        self.y_train = np.array([1.5, 3.0, 4.5, 6.0])
        self.X_test = np.array([[5], [6]])
        self.y_test = np.array([7.5, 9])

    def test_multiple_linear_regression(self) -> None:
        """Test the Multiple Linear Regression model."""
        model = MultipleLinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        expected_predictions = np.array([7.5, 9])

        np.testing.assert_almost_equal(
            predictions, expected_predictions, decimal=5)

    def test_lasso_regression(self) -> None:
        """Test the Lasso Regression model."""
        model = Lasso(alpha=0.1)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(predictions.shape, self.y_test.shape)
        self.assertTrue(np.allclose(predictions, self.y_test, atol=1))

    def test_support_vector_regression(self) -> None:
        """Test the Support Vector Regression model."""
        model = SupportVectorRegression(kernel='rbf')
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(predictions.shape, self.y_test.shape)
        self.assertTrue(np.allclose(predictions, self.y_test, atol=1))


if __name__ == "__main__":
    unittest.main()
