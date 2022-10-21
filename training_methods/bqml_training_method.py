from typing import Any, Dict

from google.cloud import bigquery

import utils
from models import dataset
from training_methods import training_method


class BQMLARIMAPlusTrainingMethod(training_method.TrainingMethod):
    """Used to run a BQML ARIMAPlus training job"""

    @property
    def id(self) -> str:
        """A unique id representing this training method.

        Returns:
            str: The id
        """
        return "bqml_arimaplus"

    @property
    def display_name(self) -> str:
        """A display_name representing this training method.

        Returns:
            str: The name
        """
        return "BQML ARIMA+"

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

        time_column = model_parameters.get("timeColumn")
        target_column = model_parameters.get("targetColumn")
        time_series_id_column = model_parameters.get("timeSeriesIdentifierColumn")
        forecast_horizon = prediction_parameters.get("forecastHorizon")

        if time_column is None:
            raise ValueError(f"Missing argument: timeColumn")

        if target_column is None:
            raise ValueError(f"Missing argument: targetColumn")

        if time_series_id_column is None:
            raise ValueError(f"Missing argument: timeSeriesIdentifierColumn")

        # Start training
        query_job = self._train(
            dataset=dataset,
            time_column=time_column,
            target_column=target_column,
            time_series_id_column=time_series_id_column,
            forecast_horizon=forecast_horizon,
        )

        # Wait for result
        _ = query_job.result()

        return str(query_job.destination)

    def evaluate(self, model: str) -> str:
        """Evaluate a model and return the BigQuery URI to its evaluation table.

        Args:
            model (str): Model to evaluate.

        Returns:
            str: The BigQuery evaluation table URI.
        """

        query_job = self._evaluate(model=model)

        # Wait for result
        _ = query_job.result()

        return str(query_job.destination)

    def predict(self, model: str, parameters: Dict[str, Any]) -> str:
        """Predict using a model and return the BigQuery URI to its prediction table.

        Args:
            model (str): Model to evaluate.
            parameters (Dict[str, Any]): The prediction parameters.

        Returns:
            str: The BigQuery prediction table URI.
        """
        query_job = self._predict(model=model, parameters=parameters)

        # Wait for result
        _ = query_job.result()

        return str(query_job.destination)

    def _train(
        self,
        dataset: dataset.Dataset,
        time_column: str,
        target_column: str,
        time_series_id_column: str,
        forecast_horizon: int,
    ) -> bigquery.QueryJob:
        client = bigquery.Client()
        project_id = client.project
        dataset_id = utils.generate_uuid()

        # Create training dataset in default region
        bq_dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        bq_dataset = client.create_dataset(bq_dataset, exists_ok=True)

        bigquery_uri = dataset.get_bigquery_uri(time_column=time_column)

        query = f"""
            CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.bqml_arima`
            OPTIONS
            (MODEL_TYPE = 'ARIMA_PLUS',
            TIME_SERIES_TIMESTAMP_COL = '{time_column}',
            TIME_SERIES_DATA_COL = '{target_column}',
            TIME_SERIES_ID_COL = '{time_series_id_column}',
            HORIZON = {forecast_horizon}
            ) AS
            SELECT
            {time_column},
            {target_column},
            {time_series_id_column}
            FROM
            `{bigquery_uri}`
        """

        # Start the query job
        return client.query(query)

    def _evaluate(
        self,
        model: str,
    ) -> bigquery.QueryJob:
        client = bigquery.Client()

        query = f"""
            SELECT
            *
            FROM
            ML.ARIMA_EVALUATE(MODEL `{model}`)
        """

        # Start the query job
        return client.query(query)

    def _predict(self, model: str, parameters: Dict[str, Any]) -> bigquery.QueryJob:
        forecast_horizon = parameters.get("forecastHorizon")

        if forecast_horizon is None:
            raise ValueError("forecastHorizon was not provided")

        client = bigquery.Client()

        query = f"""
            SELECT
            *
            FROM
            ML.FORECAST(MODEL `{model}`,
                        STRUCT({forecast_horizon} AS horizon, 0.8 AS confidence_level))            
        """

        # Start the query job
        return client.query(query)
