import abc
import dataclasses
import uuid
from datetime import datetime
from functools import cached_property
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from google.cloud import bigquery

import utils


class Dataset(abc.ABC):
    id: str
    description: str
    display_name: str
    time_column: str
    icon: Optional[str]
    recommended_model_parameters: Optional[Dict[str, Dict[str, Any]]]
    recommended_prediction_parameters: Optional[Dict[str, Dict[str, Any]]]
    test_percentage: int=0.2

    @property
    @abc.abstractmethod
    def df(self) -> pd.DataFrame:
        """The full dataset dataframe, which can be quite large.

        Returns:
            pd.DataFrame: The dataset represented as a Pandas dataframe.
        """
        pass

    @cached_property
    def columns(self) -> List[str]:
        return self.df.columns.tolist()

    @cached_property
    def df_preview(self) -> pd.DataFrame:
        return self.df.head()

    @cached_property
    def start_date(self) -> datetime:
        df = self.df
        time_values = pd.to_datetime(df[self.time_column])

        return time_values.min()

    @cached_property
    def end_date(self) -> datetime:
        df = self.df
        time_values = pd.to_datetime(df[self.time_column])

        return time_values.max()

    def as_response(self) -> Dict:
        df_preview = self.df_preview.fillna("").sort_values(self.time_column)
        df_preview["id"] = df_preview.index

        return {
            "id": self.id,
            "displayName": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "startDate": self.start_date.strftime("%m/%d/%Y"),
            "endDate": self.end_date.strftime("%m/%d/%Y"),
            "columns": self.columns,
            "dfPreview": df_preview.to_dict("records"),
            "recommendedModelParameters": self.recommended_model_parameters,
            "recommendedPredictionParameters": self.recommended_prediction_parameters,
        }

    def get_bigquery_uri(
        self,
        time_column: str,
        dataset_portion: str
    ) -> str:
      """_summary_

      Args:
          time_column (str): _description_
          dataset_portion (str): `test` or `train`

      Returns:
          str: _description_
      """

      dataset_id = utils.generate_uuid()
      table_id = utils.generate_uuid()

      # Write dataset to BigQuery table
      client = bigquery.Client()
      project_id = client.project

      bq_dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
      bq_dataset = client.create_dataset(bq_dataset, exists_ok=True)

      job_config = bigquery.LoadJobConfig(
          # Specify a (partial) schema. All columns are always written to the
          # table. The schema is used to assist in data type definitions.
          schema=[
              bigquery.SchemaField(time_column, bigquery.enums.SqlTypeNames.DATE),
          ],
          # Optionally, set the write disposition. BigQuery appends loaded rows
          # to an existing table by default, but with WRITE_TRUNCATE write
          # disposition it replaces the table with the loaded data.
          write_disposition="WRITE_TRUNCATE",
      )

      # Reference: https://cloud.google.com/bigquery/docs/samples/bigquery-load-table-dataframe
      job = client.load_table_from_dataframe(
          dataframe=self.df,
          destination=f"{project_id}.{dataset_id}.{table_id}",
          job_config=job_config,
      )  # Make an API request.

      _ = job.result()  # Wait for the job to complete.

      return str(job.destination)


@dataclasses.dataclass
class CSVDataset(Dataset):
    filepath_or_buffer: Union[str, StringIO]
    display_name: str
    time_column: str
    description: str
    icon: Optional[str] = None
    recommended_model_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    recommended_prediction_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    id: str = dataclasses.field(default_factory=utils.generate_uuid)

    @cached_property
    def df(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath_or_buffer)
        df[self.time_column] = pd.to_datetime(df[self.time_column], utc=True)
        return df


@dataclasses.dataclass
class VertexAIDataset(Dataset):
    id: str
    display_name: str
    time_column: str
    description: str
    project: str
    region: str
    icon: Optional[str] = None
    recommended_model_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    recommended_prediction_parameters: Optional[Dict[str, Dict[str, Any]]] = None

    @cached_property
    def df(self) -> pd.DataFrame:
        # TODO
        return pd.DataFrame()
