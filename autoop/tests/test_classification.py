import unittest
import numpy as np
import sys
import os
# flake8: noqa: E402
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../')))
from autoop.core.ml.model.classification.classification import (
    KNearestNeighbors, LogisticRegression, DecisionTree)


class TestClassification(unittest.TestCase):

    def setUp(self):
        """Sample training and testing data for the models."""
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
        self.y_train = np.array([0, 1, 0, 1])
        self.X_test = np.array([[2, 3], [5, 6]])
        self.y_test = np.array([1, 1])

    def test_knn(self):
        """Test the K-Nearest Neighbors (KNN) model."""
        model = KNearestNeighbors(k=3)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        expected = [0, 1]
        self.assertEqual(predictions, expected)

    def test_logistic_regression(self):
        """Test the Logistic Regression model."""
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(predictions.shape, self.y_test.shape)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_decision_tree(self):
        """Test the Decision Tree model."""
        model = DecisionTree(max_depth=3)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(predictions.shape, self.y_test.shape)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))


if __name__ == '__main__':
    unittest.main()
