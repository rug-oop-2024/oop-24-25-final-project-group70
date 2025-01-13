from abc import ABC, abstractmethod
import numpy as np

# Define available metrics
METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "mean_absolute_error",
]


def get_metric(name: str) -> "Metric":
    """
    Factory function to get a metric by name.

    Args:
        name (str): The name of the metric.

    Returns:
        Metric: An instance of the specified metric.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "f1_score":
        return F1Score()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    else:
        raise ValueError(f"Unknown metric name: {name}")


class Metric(ABC):
    """Abstract base class for all metrics.

    Metrics are used to evaluate the performance of a model by comparing
    predictions to ground truth values. Each metric class implements a
    specific evaluation metric.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the metric based on ground truth and predictions.

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The computed metric value.
        """
        pass


class Accuracy(Metric):
    """Accuracy metric for classification tasks.

    Measures the proportion of correct predictions over the total predictions.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class MeanSquaredError(Metric):
    """Mean Squared Error (MSE) metric for regression tasks.

    Calculates the average of the squared differences between predictions
    and ground truth values.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class Precision(Metric):
    """Precision metric for binary classification.

    Measures the proportion of true positives among all positive predictions.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positive = np.sum(y_pred == 1)
        if predicted_positive > 0:
            return true_positive / predicted_positive
        else:
            return 0.0


class Recall(Metric):
    """Recall metric for binary classification.

    Measures the proportion of true positives
    among all actual positive instances.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive > 0 else 0.0


class F1Score(Metric):
    """F1 Score metric for binary classification.

    Calculates the harmonic mean of precision and recall.
    Useful when dealing with imbalanced classes.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = Precision()(y_true, y_pred)
        recall = Recall()(y_true, y_pred)
        if (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0.0


class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) metric for regression tasks.

    Calculates the average of absolute differences between predictions
    and ground truth values.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
