import abc
import dataclasses
from datetime import datetime
import pandas as pd

from typing import Dict, List, Optional, Union
import uuid
from io import StringIO

from functools import cached_property


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
        return self.df.columns

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
            "columns": self.columns.tolist(),
            "df_preview": df_preview.to_dict("records"),
        }


@dataclasses.dataclass
class CSVDataset(Dataset):
    filepath_or_buffer: Union[str, StringIO]
    display_name: str
    time_column: str
    description: str
    icon: Optional[str] = None
    id: Union[uuid.UUID, None] = dataclasses.field(default_factory=uuid.uuid4)

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
