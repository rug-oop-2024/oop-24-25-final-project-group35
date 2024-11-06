from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """Defines a machine learning pipeline that includes data preprocessing,
    model training, and evaluation.

    The Pipeline class manages artifacts and processes input features,
    target feature, and metrics, ensuring the correct flow of data.
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8) -> None:
        """Initialize the Pipeline with a dataset, model, features and metrics.

        Args:
            metrics (List[Metric]): A list of metrics to evaluate the model.
            dataset (Dataset): The dataset used in the pipeline.
            model (Model): The machine learning model used for training.
            input_features (List[Feature]): List of input features.
            target_feature (Feature): The target feature for prediction.
            split (float): The train-test split ratio. Defaults to 0.8.

        Raises:
            ValueError: If there is a mismatch between the target feature
            type and the model type.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

        if (target_feature.type == "categoricals" and
                model.type != "classification"):

            raise ValueError(
                "Model must be classification for categorical target feature"
            )

        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """Returns a string representation of the pipeline configuration.

        Returns:
            str: A formatted string containing model, features, split
            and metrics.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """Provides access to the model used in the pipeline.

        Returns:
            Model: The machine learning model.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Generates and returns artifacts produced during pipeline execution.

        Returns:
            List[Artifact]: A list of artifacts generated by the pipeline,
            including preprocessing encoders and the model artifact.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config", data=pickle.dumps(
            pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact) -> None:
        """Registers an artifact with a specified name.

        Args:
            name (str): The name of the artifact.
            artifact: The artifact object to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocesses the input and target features and registers artifacts.
        """
        df = self._dataset.read()

        if self._target_feature.type == 'categorical':
            target_data = df[self._target_feature.name].values
            self._output_vector = target_data
        else:
            (target_feature_name, target_data, artifact
             ) = preprocess_features([self._target_feature], self._dataset)[0]
            self._register_artifact(target_feature_name, artifact)
            self._output_vector = target_data.flatten()

        input_results = preprocess_features(
            self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        self._input_vectors = [data for (
            feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """Splits the data into training and testing sets based
        on the split ratio."""
        split = self._split
        self._train_X = [vector[
            :int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[
            int(split * len(vector)):] for vector in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Concatenates a list of numpy arrays into a single array.

        Args:
            vectors (List[np.array]): A list of numpy arrays.

        Returns:
            np.array: A concatenated numpy array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Trains the model on the preprocessed and split training data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluates the model on the test data and calculates metrics."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_on_training_set(self) -> None:
        """Evaluates the model on the training data for training metrics."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._train_metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._train_metrics_results.append((metric, result))

    def execute(self) -> dict:
        """Executes the pipeline to perform preprocessing, splitting, training,
        and evaluation.

        Returns:
            dict: A dictionary containing training metrics, test metrics,
            and model predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate_on_training_set()
        self._evaluate()

        return {
            "train_metrics": self._train_metrics_results,
            "test_metrics": self._metrics_results,
            "predictions": self._predictions,
        }
