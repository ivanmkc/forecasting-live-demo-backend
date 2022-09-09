import abc
import dataclasses
import multiprocessing
import logging
import os
import time
from typing import Dict, List

from models import dataset, forecast
from services import training_service

import utils

NUM_PROCESSES = multiprocessing.cpu_count() - 1


@dataclasses.dataclass
class TrainingJobManagerRequest:
    training_method: str
    dataset: dataset.Dataset
    parameters: Dict
    id: str = dataclasses.field(default_factory=utils.generate_uuid)


def create_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("process.log")
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def process_tasks(
    training_service: training_service.TrainingService,
    task_queue: multiprocessing.Queue,
    completed_forecasts: multiprocessing.Queue,
):
    """
    Function for worker to poll for tasks and run them

    args:
        training_service: The training service to delegate to.
        task_queue: Mutable, process-safe queue to retrieve jobs from.
        completed_forecasts: Mutable, process-safe queue to write resultant forecasts to.
    """
    logger = create_logger()
    # proc = os.getpid()
    while True:
        if not task_queue.empty():
            try:
                request: TrainingJobManagerRequest = task_queue.get()

                if request:
                    logger.info("Processing request: {request}")
                    output_forecast = training_service.run(
                        training_method_name=request.training_method,
                        dataset=request.dataset,
                        parameters=request.parameters,
                    )

                    completed_forecasts.put(output_forecast)
            except Exception as e:
                logger.error(e)


class TrainingJobManager(abc.ABC):
    @abc.abstractmethod
    def enqueue(self, request: TrainingJobManagerRequest) -> str:
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

    _pending_jobs: multiprocessing.Queue = multiprocessing.Queue()
    _completed_forecasts: multiprocessing.Queue

    def __init__(self, training_service: training_service.TrainingService) -> None:
        super().__init__()
        self._training_service = training_service
        # self._pending_jobs: queue.Queue[TrainingJobManagerRequest] = queue.Queue()

        # TODO: Start worker to process queue
        self._start_workers()

    def _start_workers(self):
        processes = []
        print(f"Running with {NUM_PROCESSES} processes!")
        with multiprocessing.Manager() as manager:
            self._completed_forecasts = manager.Queue()

            with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                pool.apply_async(
                    process_tasks,
                    (
                        self._training_service,
                        self._pending_jobs,
                        self._completed_forecasts,
                    ),
                )
            # for _ in range(NUM_PROCESSES):
            #     p = multiprocessing.Process(
            #         target=process_tasks,
            #         args=(
            #             self._training_service,
            #             self._pending_jobs,
            #             self._completed_forecasts,
            #         ),
            #     )
            #     processes.append(p)
            #     p.start()

    def enqueue(self, request: TrainingJobManagerRequest) -> str:
        # Store job in memory
        self._pending_jobs.put(request)

        return request.id

    def list_pending_jobs(self) -> List[TrainingJobManagerRequest]:
        # TODO: Add pagination
        return list(self._pending_jobs.queue)

    def list_completed_forecasts(self) -> List[forecast.Forecast]:
        # TODO: Add pagination
        # return list(self._completed_forecasts.queue)
        # return [res.get() for res in self._completed_forecasts]
        return []
