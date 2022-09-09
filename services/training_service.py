import abc
from datetime import datetime
from typing import Any, Dict
from models import dataset
from models import training_result
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
    ) -> training_result.TrainingResult:
        training_method = self._training_registry.get(training_method_name)

        if training_method is None:
            raise ValueError(
                f"Training method '{training_method_name}' is not supported"
            )

        # Start training
        output = training_result.TrainingResult(
            start_time=start_time,
            end_time=datetime.now(),
            model_uri=None,
            error_message=None,
        )

        try:
            # Train model
            output.model_uri = training_method.train(
                dataset=dataset, parameters=parameters
            )

            # Run evaluation
            output.evaluation_uri = training_method.evaluate(model=output.model_uri)

            # Run forecast
            output.forecast_uri = training_method.forecast(
                model=output.model_uri, parameters={}
            )
        except Exception as exception:
            output.error_message = str(exception)

        return output
