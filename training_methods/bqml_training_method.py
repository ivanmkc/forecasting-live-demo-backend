from typing import Any, Dict

from google.cloud import bigquery

import utils
from models import dataset
from training_methods import training_method


class BQMLARIMAPlusTrainingMethod(training_method.TrainingMethod):
    """Used to run a BQML ARIMAPlus training job"""

    @staticmethod
    def training_method() -> str:
        """A unique key representing this training method.

        Returns:
            str: The key
        """
        return "bqml"

    def train(self, dataset: dataset.Dataset, parameters: Dict[str, Any]) -> str:
        """Train a job and return the model URI.

        Args:
            dataset (dataset.Dataset): Input dataset.
            parameters (Dict[str, Any]): The model training parameters.

        Returns:
            str: The model URI
        """

        time_column = parameters.get("timeColumn")
        target_column = parameters.get("targetColumn")
        time_series_id_column = parameters.get("timeSeriesIdentifierColumn")

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
    ) -> bigquery.QueryJob:
        client = bigquery.Client()
        project_id = client.project
        dataset_id = utils.generate_uuid()

        # Create training dataset in default region
        bq_dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        bq_dataset = client.create_dataset(bq_dataset, exists_ok=True)

        bigquery_uri = dataset.get_bigquery_uri(time_column=time_column)

        query = f"""
            create or replace model `{project_id}.{dataset_id}.bqml_arima`
            options
            (model_type = 'ARIMA_PLUS',
            time_series_timestamp_col = '{time_column}',
            time_series_data_col = '{target_column}',
            time_series_id_col = '{time_series_id_column}'
            ) as
            select
            {time_column},
            {target_column},
            {time_series_id_column}
            from
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
