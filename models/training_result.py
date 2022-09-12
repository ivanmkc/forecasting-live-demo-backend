from datetime import datetime
from typing import Any, Dict, Optional


class ForecastJobResult:
    """
    Encapsulates the results of a train-eval-forecast job.
    """

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        model_uri: Optional[str] = None,
        evaluation_uri: Optional[Dict[str, Any]] = None,
        prediction_uri: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """_summary_

        Args:
            start_time (datetime): The request start time.
            start_time (datetime): The request end time.
            model_uri (Optional[str], optional): The URI to the model. Defaults to None.
            evaluation_uri (Optional[Dict[str, Any]], optional): The BigQuery URI of the evaluation. Defaults to None.
            prediction_uri (Optional[Dict[str, Any]], optional): The BigQuery URI of the prediction. Defaults to None.
            error_message (Optional[str], optional): The error message encountered during training. Defaults to None.
        """
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.model_uri = model_uri
        self.evaluation_uri = evaluation_uri
        self.prediction_uri = prediction_uri
        self.error_message = error_message
