import abc
import dataclasses
from datetime import datetime

import pandas as pd

from models import dataset
import utils


@dataclasses.dataclass
class ForecastJob(abc.ABC):
    start_time: datetime
    status: str
    id: str = dataclasses.field(default_factory=utils.generate_uuid)


# class BQMLForecastJob(ForecastJob):
#     @property
#     def status(self) -> str:


class Forecast:
    def __init__(
        self,
        execution_date: datetime,  # Date when forecast was executed
        dataset: dataset.Dataset,  # Start date of forecast data
        df_prediction: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.execution_date = execution_date
        self.dataset = dataset
        self.df_prediction = df_prediction
