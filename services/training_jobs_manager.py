import abc
import dataclasses
import multiprocessing
import logging
import os
import time
from typing import Dict, List

from models import dataset
from services.forecasts_service import ForecastsService
from services import training_service

import utils

forecasts_service = ForecastsService()

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
    task_queue,
):
    """
    Function for worker to poll for tasks and run them
    """
    logger = create_logger()
    proc = os.getpid()
    while True:
        if not task_queue.empty():
            try:
                request: TrainingJobManagerRequest = task_queue.get()

                if request:
                    # get_word_counts(book)
                    logger.info("Processing request: {request}")
                    training_service.run_async(
                        dataset=request.dataset, **request.parameters
                    )
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

    def __init__(self, training_service: training_service.TrainingService) -> None:
        super().__init__()
        self._training_service = training_service
        # self._pending_jobs: queue.Queue[TrainingJobManagerRequest] = queue.Queue()
        self._pending_jobs: multiprocessing.Queue[
            TrainingJobManagerRequest
        ] = multiprocessing.Queue()

        # TODO: Start worker to process queue
        self._start_workers()

    def _start_workers(self):
        processes = []
        print(f"Running with {NUM_PROCESSES} processes!")
        for _ in range(NUM_PROCESSES):
            p = multiprocessing.Process(
                target=process_tasks,
                args=(
                    self._training_service,
                    self._pending_jobs,
                ),
            )
            processes.append(p)
            p.start()

    def enqueue(self, request: TrainingJobManagerRequest) -> str:
        # Store job in memory
        self._pending_jobs.put(request)

        return request.id

    def list_pending_jobs(self) -> List[TrainingJobManagerRequest]:
        # TODO: Add pagination
        return list(self._pending_jobs.queue)
