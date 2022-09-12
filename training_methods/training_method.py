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
