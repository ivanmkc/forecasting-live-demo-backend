from datetime import datetime
from typing import Optional


class Forecast:
    def __init__(
        self,
        start_time: datetime,  # Date when forecast was started
        end_time: datetime,  # Date when forecast was finished
        model_uri: Optional[str],  # Output model URI
        error_message: Optional[str],
    ) -> None:
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.model_uri = model_uri
        self.error_message = error_message
