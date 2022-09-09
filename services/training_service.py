import abc
from datetime import datetime
from typing import Any, Dict
from models import dataset
from models import forecast
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
        start_time: datetime,
        dataset: dataset.Dataset,
        parameters: Dict[str, Any],
    ) -> forecast.Forecast:
        training_method = self._training_registry.get(training_method_name)

        if training_method is None:
            raise ValueError(
                f"Training method '{training_method_name}' is not supported"
            )

        return training_method.run(
            start_time=start_time, dataset=dataset, parameters=parameters
        )
