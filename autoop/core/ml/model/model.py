from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from typing import Literal

class Model(Artifact, ABC):
    def __init__(self, name: str, model_type: Literal['classification', 'regression'], **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.model_type = model_type
        self.parameters = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save_model(self):
        self.save(new_data=np.array(list(self.parameters.values())).tobytes())

    def load_model(self):
        params_data = self.read()
        if params_data:
            self.parameters = np.frombuffer(params_data, dtype=np.float64)
        else:
            raise ValueError("No model data found to load.")

    def __repr__(self):
        return f"Model(name={self.name}, model_type={self.model_type}, parameters={self.parameters})"

    
