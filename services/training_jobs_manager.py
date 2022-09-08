import abc
import dataclasses
from typing import Collection, Dict, List

from models import dataset, forecast
from services.forecasts_service import ForecastsService
from services import training_service
import queue

import utils

forecasts_service = ForecastsService()


@dataclasses.dataclass
class TrainingJobManagerRequest:
    dataset: dataset.Dataset
    parameters: Dict
    id: str = dataclasses.field(default_factory=utils.generate_uuid)


class TrainingJobManager(abc.ABC):
    @abc.abstractmethod
    def queue(self, request: TrainingJobManagerRequest) -> str:
        pass

    @abc.abstractmethod
    def list_pending_jobs(self) -> List[TrainingJobManagerRequest]:
        # TODO: Add pagination
        pass


class MemoryTrainingJobManager(TrainingJobManager):
    """
    A job manager to queue jobs and delegate jobs to workers.

    Used for development and testing.
    """

    def __init__(self, training_service: training_service.TrainingService) -> None:
        super().__init__()
        self._training_service = training_service
        self._pending_jobs: queue.Queue[TrainingJobManagerRequest] = queue.Queue()

        # TODO: Start worker to process queue

    def queue(self, request: TrainingJobManagerRequest) -> str:
        # Store job in memory
        self._pending_jobs.put(request)

        return request.id

    def list_pending_jobs(self) -> List[TrainingJobManagerRequest]:
        # TODO: Add pagination
        return list(self._pending_jobs.queue)
