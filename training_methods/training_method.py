import abc
from typing import Any, Dict

from models import dataset, forecast_job_request


class TrainingMethod(abc.ABC):
    @property
    @abc.abstractmethod
    def id(self) -> str:
        """A unique id representing this training method.

        Returns:
            str: The id
        """
        pass

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """A display_name representing this training method.

        Returns:
            str: The name
        """
        pass

    @abc.abstractmethod
    def dataset_time_series_identifier_column(
        self, job_request: forecast_job_request.ForecastJobRequest
    ) -> str:
        """The column representing the time series identifier variable in the dataset dataframe.

        Returns:
            str: The column name
        """
        pass

    @abc.abstractmethod
    def dataset_time_column(
        self, job_request: forecast_job_request.ForecastJobRequest
    ) -> str:
        """The column representing the time variable in the dataset dataframe.

        Returns:
            str: The column name
        """
        pass

    @abc.abstractmethod
    def dataset_target_column(
        self, job_request: forecast_job_request.ForecastJobRequest
    ) -> str:
        """The column representing the target variable in the dataset dataframe.

        Returns:
            str: The column name
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def evaluate(self, model: str) -> str:
        """Evaluate a model and return the BigQuery table ID to its evaluation
        table.

        Args:
            model (str): Model to evaluate.

        Returns:
            str: The BigQuery evaluation table ID.
        """
        pass

    @abc.abstractmethod
    def predict(
        self,
        model: str,
        model_parameters: Dict[str, Any],
        prediction_parameters: Dict[str, Any],
    ) -> str:
        """Predict using a model and return the BigQuery table ID to its
        prediction table.

        Args:
            model (str): Model to evaluate.
            model_parameters (Dict[str, Any]): The model training parameters.
            prediction_parameters (Dict[str, Any]): The prediction parameters.

        Returns:
            str: The BigQuery prediction table ID.
        """
        pass
