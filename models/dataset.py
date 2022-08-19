import abc
import dataclasses
from datetime import datetime
import pandas as pd

from typing import Dict, List, Union
import uuid
from io import StringIO


class Dataset(abc.ABC):
    id: str
    display_name: str
    time_column: str

    @property
    @abc.abstractmethod
    def df(self) -> pd.DataFrame:
        pass

    @property
    def columns(self) -> List[str]:
        return self.df.columns

    @property
    def df_preview(self) -> pd.DataFrame:
        return self.df.head()

    @property
    def start_date(self) -> datetime:
        df = self.df
        time_values = pd.to_datetime(df[self.time_column])

        return time_values.min()

    def to_dict(self) -> Dict:
        return {
            "start_date": self.start_date.strftime("%m/%d/%Y, %H:%M:%S"),
            "columns": self.columns.tolist(),
            "df_preview": self.df_preview.fillna("").to_dict(),
        }


@dataclasses.dataclass
class CSVDataset(Dataset):
    filepath_or_buffer: Union[str, StringIO]
    display_name: str
    time_column: str
    id: Union[uuid.UUID, None] = dataclasses.field(default_factory=uuid.uuid4)

    @property
    def df(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath_or_buffer)
        return df.head()


@dataclasses.dataclass
class VertexAIDataset(Dataset):
    id: str
    display_name: str
    time_column: str
    project: str
    region: str

    @property
    def df(self) -> pd.DataFrame:
        # TODO
        return pd.DataFrame()
