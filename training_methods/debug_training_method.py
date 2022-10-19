from typing import Any, Dict

from google.cloud import bigquery

import time

import utils
from models import dataset
from training_methods import training_method

MAX_DELAY_IN_SECONDS = 60


class DebugTrainingMethod(training_method.TrainingMethod):
    """Used to run a dummy training job for integration testing as the actual jobs can take a long time.

    It can wait a specified number of seconds or error out, depending on the parameters passed to it.
    """

    @staticmethod
    def training_method() -> str:
        """A unique key representing this training method.

        Returns:
            str: The key
        """

        return "debug"

    def train(
        self,
        dataset: dataset.Dataset,
        model_parameters: Dict[str, Any],
        prediction_parameters: Dict[str, Any],
    ) -> str:
        """Train a job and return the model URI.

        Args:
            dataset (dataset.Dataset): Input dataset.
            model_parameters (Dict[str, Any]): The model training parameters.
            prediction_parameters (Dict[str, Any]): The prediction parameters.

        Returns:
            str: The model URI
        """

        # Sleep for a specified delay, not more than a max.
        delay_in_seconds = min(
            model_parameters.get("delay_in_seconds", 5), MAX_DELAY_IN_SECONDS
        )

        time.sleep(delay_in_seconds)

        error_message = model_parameters.get("error_message")

        if error_message:
            raise ValueError(error_message)

        return "debug.model"

    def evaluate(self, model: str) -> str:
        """Evaluate a model and return the BigQuery URI to its evaluation table.

        Args:
            model (str): Model to evaluate.

        Returns:
            str: The BigQuery evaluation table URI.
        """

        return "debug.evaluation"

    def predict(self, model: str, parameters: Dict[str, Any]) -> str:
        """Predict using a model and return the BigQuery URI to its prediction table.

        Args:
            model (str): Model to evaluate.
            parameters (Dict[str, Any]): The prediction parameters.

        Returns:
            str: The BigQuery prediction table URI.
        """

        return "debug.prediction"