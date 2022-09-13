import dataclasses
from datetime import datetime
from typing import Any, Dict
import utils

from models import dataset


@dataclasses.dataclass
class ForecastJobRequest:
    """An encapsulation of the training job request"""

    # The unique key associated with a training method.
    training_method: str

    # The dataset used for model training.
    dataset: dataset.Dataset

    # The request start time.
    start_time: datetime

    # Parameters for training.
    model_parameters: Dict[str, Any]

    # Parameters for prediction.
    prediction_parameters: Dict[str, Any]

    # The unique request id.
    id: str = dataclasses.field(default_factory=utils.generate_uuid)
