import abc

from datetime import datetime
from google.cloud import bigquery

import utils
from models import dataset, forecast
from typing import Any, Dict


class TrainingMethod(abc.ABC):
    @abc.abstractmethod
    def run(
        self, start_time: datetime, dataset: dataset.Dataset, parameters: Dict[str, Any]
    ) -> str:
        """
        Train a job and return a job ID
        """
        pass


class BQMLARIMAPlusTrainingMethod(TrainingMethod):
    def run(
        self, start_time: datetime, dataset: dataset.Dataset, parameters: Dict[str, Any]
    ) -> str:
        """
        Train a job and return a job ID
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
        output_forecast = forecast.Forecast(
            start_time=start_time,
            end_time=datetime.now(),
            model_uri=None,
            error_message=None,
        )

        try:
            query_job = self._train_bigquery(
                dataset=dataset,
                time_column=time_column,
                target_column=target_column,
                time_series_id_column=time_series_id_column,
            )

            # Wait for result
            _ = query_job.result()

            output_forecast.model_uri = str(query_job.destination)
        except Exception as exception:
            output_forecast.error_message = str(exception)

        return output_forecast

    # This has to be thread-safe
    def _train_bigquery(
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
