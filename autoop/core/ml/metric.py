from abc import ABC, abstractmethod
import numpy as np

# List of supported metric names
METRICS = [
    "mean_squared_error",
    "accuracy",
    "cohens_kappa",
    "mean_absolute_error",
    "r2",
    "CSI"
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
    elif name == "CSI":
        return CSI()
    else:
        raise ValueError(f"Metric {name} is not supported. "
                         "Available metrics: {METRICS}")


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

        all_labels = np.unique(np.concatenate((y_true, y_pred)))
        label_to_index = {label: idx for idx, label in enumerate(all_labels)}
        y_true_int = np.array([label_to_index[label] for label in y_true])
        y_pred_int = np.array([label_to_index[label] for label in y_pred])

        class_counts = np.bincount(y_true_int, minlength=len(all_labels))
        pred_counts = np.bincount(y_pred_int, minlength=len(all_labels))

        expected_agreement = np.sum((class_counts / total) * (
            pred_counts / total))

        kappa = (observed_agreement - expected_agreement) / (
            1 - expected_agreement)
        return kappa


class CSI(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        all_labels = np.unique(np.concatenate((y_true, y_pred)))
        label_to_index = {label: idx for idx, label in enumerate(all_labels)}
        y_true_int = np.array([label_to_index[label] for label in y_true])
        y_pred_int = np.array([label_to_index[label] for label in y_pred])

        num_classes = len(all_labels)
        csi_scores = []

        for i in range(num_classes):
            true_positive = np.sum((y_true_int == i) & (y_pred_int == i))
            false_positive = np.sum((y_true_int != i) & (y_pred_int == i))
            false_negative = np.sum((y_true_int == i) & (y_pred_int != i))
            denominator = true_positive + false_positive + false_negative
            if denominator == 0:
                csi = 0.0
            else:
                csi = true_positive / denominator
            csi_scores.append(csi)

        mean_csi = np.mean(csi_scores)
        return mean_csi


class MeanAbsoluteError(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mae = np.mean(np.abs(y_true - y_pred))
        return mae


class R2(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R-squared (coefficient of determination) value.

        Args:
            y_true (np.ndarray): The true values of the target variable.
            y_pred (np.ndarray): The predicted values from the model.

        Returns:
            float: The R-squared value.
        """
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
