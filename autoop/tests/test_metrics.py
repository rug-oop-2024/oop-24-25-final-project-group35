import unittest
import numpy as np
import sys
import os
# flake8: noqa: E402
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from autoop.core.ml.metric import (
    get_metric,
    METRICS,
    Metric,
    MeanSquaredError,
    Accuracy,
    CohensKappa,
    CSI,
    MeanAbsoluteError,
    R2
)

class TestMetrics(unittest.TestCase):

    def setUp(self) -> None:
        """Example data for testing."""
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 2.1, 3.2, 4.2, 5.3])
        self.y_class_true = np.array([1, 0, 1, 1, 0])
        self.y_class_pred = np.array([1, 0, 1, 0, 0])

    def test_get_metric(self) -> None:
        """Test that the get_metric function returns the correct instances."""
        for metric_name in METRICS:
            metric = get_metric(metric_name)
            self.assertTrue(isinstance(metric, Metric))

        with self.assertRaises(ValueError):
            get_metric("unsupported_metric")

    def test_mean_squared_error(self) -> None:
        """Test the MeanSquaredError calculation."""
        mse = MeanSquaredError()
        result = mse(self.y_true, self.y_pred)
        expected = np.mean((self.y_true - self.y_pred) ** 2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_mean_absolute_error(self) -> None:
        """Test the MeanAbsoluteError calculation."""
        mae = MeanAbsoluteError()
        result = mae(self.y_true, self.y_pred)
        expected = np.mean(np.abs(self.y_true - self.y_pred))
        self.assertAlmostEqual(result, expected, places=5)

    def test_accuracy(self) -> None:
        """Test the Accuracy calculation."""
        accuracy = Accuracy()
        result = accuracy(self.y_class_true, self.y_class_pred)
        expected = np.sum(
            self.y_class_true == self.y_class_pred) / len(self.y_class_true)
        self.assertAlmostEqual(result, expected, places=5)

    def test_r2(self) -> None:
        """Test the R2 calculation."""
        r2 = R2()
        result = r2(self.y_true, self.y_pred)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        expected = 1 - (ss_res / ss_tot)
        self.assertAlmostEqual(result, expected, places=5)

    def test_cohens_kappa(self) -> None:
        """Test the CohensKappa calculation."""
        kappa = CohensKappa()
        result = kappa(self.y_class_true, self.y_class_pred)
        self.assertTrue(-1 <= result <= 1)

    def test_csi(self) -> None:
        """Test the CSI calculation."""
        csi = CSI()
        result = csi(self.y_class_true, self.y_class_pred)
        self.assertTrue(0 <= result <= 1)

