import abc
import dataclasses
import uuid
from datetime import datetime
from functools import cached_property
from io import StringIO
from typing import Dict, List, Optional, Union

import pandas as pd
from google.cloud import bigquery

import utils


class Dataset(abc.ABC):
    id: str
    description: str
    display_name: str
    time_column: str
    icon: Optional[str]

    @property
    @abc.abstractmethod
    def df(self) -> pd.DataFrame:
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

    def to_dict(self) -> Dict:
        df_preview = self.df_preview.fillna("").sort_values(self.time_column)
        df_preview["id"] = df_preview.index

        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "start_date": self.start_date.strftime("%m/%d/%Y"),
            "end_date": self.end_date.strftime("%m/%d/%Y"),
            "columns": self.columns,
            "df_preview": df_preview.to_dict("records"),
        }

    @cached_property
    def bigquery_uri(self) -> str:
        dataset_id = utils.generate_uuid()
        table_id = utils.generate_uuid()

        # Write dataset to BigQuery table
        client = bigquery.Client()
        project_id = client.project

        bq_dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        bq_dataset = client.create_dataset(bq_dataset, exists_ok=True)

        # Reference: https://cloud.google.com/bigquery/docs/samples/bigquery-load-table-dataframe
        job = client.load_table_from_dataframe(
            self.df, f"{project_id}.{dataset_id}.{table_id}"
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
    id: str = dataclasses.field(default_factory=utils.generate_uuid)

    @cached_property
    def df(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath_or_buffer)
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

    @cached_property
    def df(self) -> pd.DataFrame:
        # TODO
        return pd.DataFrame()
