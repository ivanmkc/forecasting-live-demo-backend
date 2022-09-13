import abc
from typing import Any, Dict

from models import dataset


class TrainingMethod(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def training_method() -> str:
        """A unique key representing this training method.

        Returns:
            str: The key
        """
        pass

    @abc.abstractmethod
    def train(self, dataset: dataset.Dataset, parameters: Dict[str, Any]) -> str:
        """Train a job and return the model URI.

        Args:
            dataset (dataset.Dataset): Input dataset.
            parameters (Dict[str, Any]): The model training parameters.

        Returns:
            str: The model URI
        """
        pass

    @abc.abstractmethod
    def evaluate(self, model: str) -> str:
        """Evaluate a model and return the BigQuery URI to its evaluation table.

        Args:
            model (str): Model to evaluate.

        Returns:
            str: The BigQuery evaluation table URI.
        """
        pass

    @abc.abstractmethod
    def predict(self, model: str, parameters: Dict[str, Any]) -> str:
        """Predict using a model and return the BigQuery URI to its prediction table.

        Args:
            model (str): Model to evaluate.
            parameters (Dict[str, Any]): The prediction parameters.

        Returns:
            str: The BigQuery prediction table URI.
        """
        pass
