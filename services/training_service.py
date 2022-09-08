import abc
from typing import Dict, List
from services.forecasts_service import ForecastsService
from models import dataset
from training_methods import training_method


class TrainingService:
    def __init__(
        self, training_registry: Dict[str, training_method.TrainingMethod]
    ) -> None:
        super().__init__()

        # TODO: Register training methods
        self._training_registry = training_registry

    def run_async(self, dataset: dataset.Dataset, **kwargs):
        # TODO: Add pagination
        pass
