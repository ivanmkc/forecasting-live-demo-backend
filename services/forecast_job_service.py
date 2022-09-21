import abc
from datetime import datetime
from typing import Any, Dict

from models import dataset, forecast_job_result
from models.forecast_job_request import ForecastJobRequest
from training_methods import training_method


class ForecastJobService:
    """
    This service handles model training, evaluation and prediction.
    """

    def __init__(
        self, training_registry: Dict[str, training_method.TrainingMethod]
    ) -> None:
        """_summary_

        Args:
            training_registry (Dict[str, training_method.TrainingMethod]): _description_
        """
        super().__init__()

        # TODO: Register training methods
        self._training_registry = training_registry

    def run(self, request: ForecastJobRequest) -> forecast_job_result.ForecastJobResult:
        """Run model training, evaluation and prediction for a given `training_method_name`. Waits for completion.

        Args:
            training_method_name (str): The training method name as defined in the training registry.
            start_time (datetime): Start time of job.
            dataset (dataset.Dataset): The dataset used for training.
            model_parameters (Dict[str, Any]): The parameters for training.
            prediction_parameters (Dict[str, Any]): The paramters for prediction.

        Raises:
            ValueError: Any error that happens during training, evaluation or prediction.

        Returns:
            forecast_job_result.ForecastJobResult: The results containing the URIs for each step.
        """
        training_method = self._training_registry.get(request.training_method_name)

        if training_method is None:
            raise ValueError(
                f"Training method '{request.training_method_name}' is not supported"
            )

        # Start training
        output = forecast_job_result.ForecastJobResult(
            end_time=datetime.now(),
            request=request,
            model_uri=None,
            error_message=None,
        )

        try:
            # Train model
            output.model_uri = training_method.train(
                dataset=request.dataset, parameters=request.model_parameters
            )

            # Run evaluation
            output.evaluation_uri = training_method.evaluate(model=output.model_uri)

            # Run prediction
            output.prediction_uri = training_method.predict(
                model=output.model_uri, parameters=request.prediction_parameters
            )
        except Exception as exception:
            output.error_message = str(exception)

        return output
