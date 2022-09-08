import abc
from typing import Any, Dict, List
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

    def run(
        self,
        training_method_name: str,
        dataset: dataset.Dataset,
        parameters: Dict[str, Any],
    ):
        training_method = self._training_registry.get(training_method_name)

        if training_method is None:
            raise ValueError(
                f"Training method '{training_method_name}' is not supported"
            )

        training_method.run(dataset=dataset, parameters=parameters)
