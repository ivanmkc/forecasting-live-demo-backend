import abc
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from google.cloud import bigquery

import utils
from models import dataset, training_result


class TrainingMethod(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def training_method() -> str:
        pass

    @abc.abstractmethod
    def train(self, dataset: dataset.Dataset, parameters: Dict[str, Any]) -> str:
        """
        Train a job and return the model URI.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, model: str) -> str:
        """
        Evaluate the model and return the table URI.
        """
        pass

    @abc.abstractmethod
    def forecast(self, model: str, parameters: Dict[str, Any]) -> str:
        """
        Run the forecast and return the table URI.
        """
        pass


class BQMLARIMAPlusTrainingMethod(TrainingMethod):
    @staticmethod
    def training_method() -> str:
        return "bqml"

    def train(self, dataset: dataset.Dataset, parameters: Dict[str, Any]) -> str:
        """
        Train a job and return the results.
        """

        time_column = parameters.get("time_column")
        target_column = parameters.get("target_column")
        time_series_id_column = parameters.get("time_series_id_column")

        if time_column is None:
            raise ValueError(f"Missing argument: time_column")

        if target_column is None:
            raise ValueError(f"Missing argument: target_column")

        if time_series_id_column is None:
            raise ValueError(f"Missing argument: time_column")

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
        """
        Run evaluation and return the BigQuery table.
        """

        query_job = self._evaluate(model=model)

        # Wait for result
        _ = query_job.result()

        return str(query_job.destination)

    def forecast(self, model: str, parameters: Dict[str, Any]) -> str:
        """
        Run the forecast and return the table URI.
        """
        query_job = self._forecast(model=model, parameters=parameters)

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

    def _forecast(self, model: str, parameters: Dict[str, Any]) -> bigquery.QueryJob:
        client = bigquery.Client()

        query = f"""
            SELECT
            *
            FROM
            ML.FORECAST(MODEL `{model}`,
                        STRUCT(30 AS horizon, 0.8 AS confidence_level))            
        """

        # Start the query job
        return client.query(query)
