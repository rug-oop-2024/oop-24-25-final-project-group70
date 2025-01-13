from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Dict, Any


class Model(ABC):
    """
    Base class for models in machine learning, mapping input features
    to a target feature.

    Attributes:
        parameters (Dict[str, Any]): Dictionary to store model parameters,
        including both
            strict parameters (for prediction) and hyperparameters
            (for training).
    """

    def __init__(self, **kwargs):
        """
        Initializes the Model with optional parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for setting model parameters.
        """
        self.parameters = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the provided data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target feature.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values based on the input features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        pass

    def save(self) -> Artifact:
        """
        Saves the model by creating an Artifact containing model parameters.

        Returns:
            Artifact: An Artifact instance representing the model's state.
        """
        return Artifact(
            asset_path="model_artifact",  # Example path, replace as needed
            version="1.0",
            data=deepcopy(self.parameters),
            metadata={"model_type": self.__class__.__name__},
            type="model",
            tags=["model", "ml"]
        )

    def load(self, artifact: Artifact) -> None:
        """
        Loads the model parameters from a given Artifact, restoring its state.

        Args:
            artifact (Artifact): An Artifact instance
            containing the model parameters.
        """
        self.parameters = deepcopy(artifact.data)

    def set_params(self, **params) -> None:
        """
        Sets model parameters, including strict parameters and hyperparameters.

        Args:
            **params: Arbitrary keyword arguments to update model parameters.
        """
        self.parameters.update(params)

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieves model parameters, including both strict
        parameters and hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary containing the model parameters.
        """
        return self.parameters
