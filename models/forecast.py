from datetime import datetime


class Forecast:
    def __init__(
        self,
        start_time: datetime,  # Date when forecast was started
        end_time: datetime,  # Date when forecast was finished
        model_uri: str,  # Output model URI
    ) -> None:
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.model_uri = model_uri
