import abc
from asyncio import Future
import dataclasses
from concurrent import futures
from datetime import datetime
from tracemalloc import start
from typing import Any, Dict, List

from models import dataset, forecast
from services import training_service

import utils


@dataclasses.dataclass
class TrainingJobManagerRequest:
    training_method: str
    dataset: dataset.Dataset
    start_time: datetime
    parameters: Dict
    id: str = dataclasses.field(default_factory=utils.generate_uuid)


class TrainingJobManager(abc.ABC):
    @abc.abstractmethod
    def enqueue(self, request: TrainingJobManagerRequest) -> str:
        pass

    @abc.abstractmethod
    def list_pending_jobs(self) -> List[Dict[str, Any]]:
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
        self._thread_pool_executor = futures.ThreadPoolExecutor()
        self._pending_jobs: Dict[str, TrainingJobManagerRequest] = {}
        self._completed_jobs: List[Dict[str, forecast.Forecast]] = []

    def _process_request(self, request: TrainingJobManagerRequest):
        output_forecast = self._training_service.run(
            training_method_name=request.training_method,
            start_time=request.start_time,
            dataset=request.dataset,
            parameters=request.parameters,
        )

        return request.id, output_forecast

    def _append_completed_forecast(self, future: Future):
        job_id, forecast = future.result()

        if forecast:
            # Clear pending job
            del self._pending_jobs[job_id]

            # Append completed forecasts
            self._completed_jobs.append({job_id: forecast})

    def enqueue(self, request: TrainingJobManagerRequest) -> str:
        self._pending_jobs[request.id] = request

        future = self._thread_pool_executor.submit(self._process_request, request)
        future.add_done_callback(self._append_completed_forecast)

        return request.id

    def list_pending_jobs(self) -> List[Dict[str, Any]]:
        # TODO: Add pagination
        return [
            {
                "job_id": request.id,
                "training_method": request.training_method,
                "dataset_id": request.dataset.id,
                "parameters": request.parameters,
                "start_time": request.start_time,
            }
            for job_id, request in self._pending_jobs.items()
        ]

    def list_completed_jobs(self) -> List[forecast.Forecast]:
        # TODO: Add pagination
        return self._completed_jobs
