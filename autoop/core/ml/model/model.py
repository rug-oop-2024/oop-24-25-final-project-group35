from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from typing import Literal
import pickle


class Model(Artifact, ABC):
    """Abstract base class for machine learning models,
    which inherits from Artifact.

    The Model class provides a structure for machine learning models,
    including abstract methods for fitting and predicting, as well as methods
    to serialize, save, and load model parameters.
    """

    def __init__(
            self, name: str, model_type: Literal[
                'classification', 'regression'], **kwargs) -> None:
        """Initializes a Model instance with a name, model type
        and additional parameters.

        Args:
            name (str): The name of the model.
            model_type (Literal['classification', 'regression']): Specifies
            whether the model is for classification or regression tasks.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._name = name
        self._model_type = model_type
        self.parameters = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Abstract method to fit the model to data.

        Args:
            X (np.ndarray): Feature matrix for training.
            y (np.ndarray): Target values for training.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Abstract method to make predictions based on input data.

        Args:
            X (np.ndarray): Feature matrix for which predictions are made.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    def to_artifact(self, name: str) -> Artifact:
        """Converts the model's state to an Artifact object for saving.

        Args:
            name (str): The name of the artifact to create.

        Returns:
            Artifact: An Artifact instance containing serialized model data.
        """
        data = pickle.dumps(self)
        return Artifact(
            name=name,
            data=data,
            type='model',
            metadata={'model_type': self._model_type},
        )

    def save_model(self) -> None:
        """Saves the model's parameters by serializing them to bytes."""
        self.save(new_data=np.array(list(self.parameters.values())).tobytes())

    def load_model(self) -> None:
        """Loads model parameters from saved data.

        Raises:
            ValueError: If no model data is found to load.
        """
        params_data = self.read()
        if params_data:
            self.parameters = np.frombuffer(params_data, dtype=np.float64)
        else:
            raise ValueError("No model data found to load.")

    def __repr__(self) -> str:
        """Provides a string representation of the Model instance.

        Returns:
            str: A formatted string showing model name, type, and parameters.
        """
        return (f"Model(name={self._name}, model_type={self._model_type}, "
                f"parameters={self.parameters})")
