import dataclasses
from datetime import datetime
from typing import Dict, List

from google.cloud import bigquery

import utils
from models import dataset, forecast
from services.forecasts_service import ForecastsService

forecasts_service = ForecastsService()

# This has to be thread-safe
def train_bigquery(
    dataset: dataset.Dataset,
    time_column: str,
    target_column: str,
    time_series_id_column: str,
) -> forecast.Forecast:
    client = bigquery.Client()
    project_id = client.project
    dataset_id = utils.generate_uuid()

    # Create training dataset in default region
    bq_dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
    bq_dataset = client.create_dataset(bq_dataset, exists_ok=True)

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
        `{dataset.bigquery_uri}`
    """

    query_job = client.query(query)

    df_prediction = query_job.to_dataframe()

    forecast = forecast.Forecast(
        execution_date=datetime.now(),
        dataset=dataset,
        df_prediction=df_prediction,
    )

    forecasts_service.append_forecast(forecast)

    return forecast
