import dataclasses
from datetime import datetime
import pandas as pd

from models import dataset


class Forecast:
    def __init__(
        self,
        execution_date: datetime,  # Date when forecast was executed
        dataset: dataset.Dataset,  # Start date of forecast data
        df_predicted: pd.DataFrame,
        df_actual: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.execution_date = execution_date
        self.dataset = dataset
        self.df_predicted = df_predicted
        self.df_actual = df_actual
