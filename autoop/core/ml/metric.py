from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# List of supported metric names
METRICS = [
    "mean_squared_error",
    "accuracy",
    "cohens_kappa",
    "mean_absolute_error",
    "r2",
    "MCC"
]

def get_metric(name: str):
    """Factory function to get a metric by name."""
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r2":
        return R2()
    elif name == "cohens_kappa":
        return CohensKappa()
    elif name == "MCC":
        return MCC()
    else:
        raise ValueError(f"Metric {name} is not supported. Available metrics: {METRICS}")

class Metric(ABC):

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class MeanSquaredError(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

class Accuracy(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        correct_predictions = np.sum(y_true == y_pred)
        accuracy = correct_predictions / len(y_true)
        return accuracy


class CohensKappa(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        total = len(y_true)
        observed_agreement = np.sum(y_true == y_pred) / total

        class_counts = np.bincount(y_true)
        pred_counts = np.bincount(y_pred)
        expected_agreement = np.sum((class_counts / total) * (pred_counts / total))

        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return kappa

class MCC(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        if denominator == 0:
            return 0.0
        return numerator / denominator


class MeanAbsoluteError(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

class R2(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        explained_variance = np.sum((y_true - y_pred) ** 2)
        if total_variance == 0:
            return 0.0
        r2 = 1 - (explained_variance / total_variance)
        return r2


y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

mse_metric = get_metric("mean_squared_error")
mae_metric = get_metric("mean_absolute_error")
r2_metric = get_metric("r2")

print(f"MSE: {mse_metric(y_true_reg, y_pred_reg)}")
print(f"MAE: {mae_metric(y_true_reg, y_pred_reg)}")
print(f"R2: {r2_metric(y_true_reg, y_pred_reg)}")


y_true_class = np.array([1, 0, 1, 1])
y_pred_class = np.array([1, 0, 0, 1])

accuracy_metric = get_metric("accuracy")
kappa_metric = get_metric("cohens_kappa")
mcc_metric = get_metric("MCC")


print(f"Accuracy: {accuracy_metric(y_true_class, y_pred_class)}")
print(f"CohensKappa: {kappa_metric(y_true_class, y_pred_class)}")
print(f"MCC: {mcc_metric(y_true_class, y_pred_class)}")