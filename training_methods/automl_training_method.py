from tokenize import Double
from typing import Any, Dict, List, Optional, ParamSpec
from uuid import UUID
import pandas as pd

from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform import models

import utils
from models import dataset
from training_methods import training_method

aiplatform.init()

class AutoMLForecastingTrainingMethod(training_method.TrainingMethod):
    @staticmethod
    def training_method() -> str:
        """A unique key representing this training method.

        Returns:
            str: The key
        """
        return "automl"

    def train(self, dataset: dataset.Dataset, parameters: Dict[str, Any]) -> str:
        """Train a job and return the model URI.

        Args:
            dataset (dataset.Dataset): Input timeseries dataset.
            parameters (Dict[str, Any]): The model training parameters.

        Returns:
            str: The model resource name
        """
        # create dataset here
        time_column = parameters.get("time_column")
        target_column = parameters.get("target_column")
        time_series_id_column = parameters.get("time_series_id_column")
        forecast_horizon = parameters.get("forecast_horizon")
        data_granularity_unit = parameters.get("data_granularity_unit")
        data_granularity_count = parameters.get("data_granularity_count")
        optimization_objective = parameters.get("optimization_objective")
        column_specs = parameters.get("column_specs")
        time_series_attribute_columns = parameters.get("time_series_attribute_columns")

        if time_column is None:
            raise ValueError(f"Missing argument: time_column")

        if target_column is None:
            raise ValueError(f"Missing argument: target_column")

        if time_series_id_column is None:
            raise ValueError(f"Missing argument: time_series_id_column")

        if forecast_horizon is None:
            raise ValueError(f"Missing argument: forecast_horizon")

        if data_granularity_unit is None:
          raise ValueError(f"Missing argument: data_granularity_unit")

        if data_granularity_count is None:
          raise ValueError(f"Missing argument: data_granularity_count")

        # Start training
        model = self._train(
            dataset=dataset,
            time_column=time_column,
            target_column=target_column,
            time_series_id_column=time_series_id_column,
            forecast_horizon=forecast_horizon,
            data_granularity_unit=data_granularity_unit,
            data_granularity_count=data_granularity_count,
            optimization_objective=optimization_objective,
            column_specs=column_specs,
            time_series_attribute_columns=time_series_attribute_columns
        )

        return model.resource_name

    def evaluate(self, model: str) -> str:
        """Evaluate a model and return the BigQuery URI to its evaluation table.

        Args:
            model (str): Model to evaluate.

        Returns:
            str: The BigQuery evaluation table URI.
        """

        table_uri = self._evaluate(model_name=model)

        return table_uri

    def predict(self, model: str, parameters: Dict[str, Any]) -> str:
        """Predict using a model and return the BigQuery URI to its prediction table.

        Args:
            model (str): Model to evaluate.
            parameters (Dict[str, Any]): The prediction parameters.

        Returns:
            str: The BigQuery prediction table URI.
        """
        bigquery_source = parameters.get("bigquery_source")

        if bigquery_source is None:
          raise ValueError("Missing argument: bigquery_source")

        job = self._predict(
          model=model,
          bigquery_source=bigquery_source
        )

        output_dataset = job.output_info.bigquery_output_dataset
        output_table = job.output_info.bigquery_output_table
        bq_output_table_uri  = f"{output_dataset}/{output_table}"

        return bq_output_table_uri

    def _train(
        self,
        dataset: dataset.Dataset,
        time_column: str,
        target_column: str,
        time_series_id_column: str,
        forecast_horizon: int,
        data_granularity_unit: str,
        data_granularity_count: int,
        optimization_objective: Optional[str] = None,
        column_specs: Optional[Dict[str, str]] = None,
        time_series_attribute_columns: Optional[List[str]] = None,
    ) -> models.Model:

        uuid = utils.generate_uuid()

        training_job = aiplatform.AutoMLForecastingTrainingJob(
          display_name=f"automl-job-{uuid}",
          optimization_objective=optimization_objective,
          column_specs=column_specs
          )

        # Start running the training pipeline
        model = training_job.run(
          dataset=dataset,
          target_column=target_column,
          time_column=time_column,
          time_series_identifier_column=time_series_id_column,
          available_at_forecast_columns=[time_column],
          unavailable_at_forecast_columns=[target_column],
          time_series_attribute_columns=time_series_attribute_columns,
          forecast_horizon=forecast_horizon,
          data_granularity_unit=data_granularity_unit,
          data_granularity_count=data_granularity_count,
          model_display_name=f"automl-{uuid}",
        )

        return model

    def _evaluate(self, model_name: str) -> str:

      # Get the model resource
      model = aiplatform.Model(model_name=model_name)

      # check if there us eval item
      if len(model.list_model_evaluations()) > 0:
        # Parse evaluation data
        model_evaluations = model.list_model_evaluations()[0].to_dict()
        evaluation_metrics = model_evaluations["metrics"]

        evaluation_metrics_df = pd.DataFrame(
          evaluation_metrics.items(),
          columns=["metric", "value"]
        )

        # Construct a BigQuery client object.
        client = bigquery.Client()
        project_id = client.project
        dataset_id = utils.generate_uuid()

        # Create evaluation dataset in default region
        bq_dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        bq_dataset = client.create_dataset(bq_dataset, exists_ok=True)

        # Create a bq table in the dataset and upload the evaluation metrics
        table_id = f"{project_id}.{dataset_id}.automl-evaluation"

        job_config = bigquery.LoadJobConfig(
            #The schema is used to assist in data type definitions.
            schema=[
                bigquery.SchemaField("metric", bigquery.enums.SqlTypeNames.STRING),
                bigquery.SchemaField("value", bigquery.enums.SqlTypeNames.FLOAT64),
            ],
            # Optionally, set the write disposition. BigQuery appends loaded rows
            # to an existing table by default, but with WRITE_TRUNCATE write
            # disposition it replaces the table with the loaded data.
            write_disposition="WRITE_TRUNCATE"
        )

        job = client.load_table_from_dataframe(
          dataframe=evaluation_metrics_df,
          destination=table_id,
          job_config=job_config,
        )
        # Wait for the job to complete.
        job.result()

        return str(job.destination)
      else:
        raise ValueError("Model evaluation data does not exist for model {model}!")


    def _predict(
      self,
      model: str,
      bigquery_source: str
    ) -> aiplatform.BatchPredictionJob:

        client = bigquery.Client()
        project_id = client.project

        model = aiplatform.Model(model_name=model)

        batch_prediction_job = model.batch_predict(
          job_display_name=f"automl_forecasting_{utils.generate_uuid()}",
          bigquery_source=bigquery_source,
          instances_format="bigquery",
          bigquery_destination_prefix=f"bq://{project_id}",
          predictions_format="bigquery",
          generate_explanation=True,
          sync=True
          )

        return batch_prediction_job


