from datetime import datetime
from typing import Any, Dict, Optional


class TrainingResult:
    def __init__(
        self,
        start_time: datetime,  # Date when training was started
        end_time: datetime,  # Date when training was finished
        model_uri: Optional[str] = None,  # Output model URI
        evaluation_uri: Optional[Dict[str, Any]] = None,
        forecast_uri: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.model_uri = model_uri
        self.evaluation_uri = evaluation_uri
        self.forecast_uri = forecast_uri
        self.error_message = error_message
