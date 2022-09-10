import abc
import dataclasses
from concurrent import futures
from datetime import datetime
from typing import Any, Awaitable, Dict, List, Optional, Tuple

import pandas as pd
from google.cloud import bigquery

import utils
from models import dataset, training_result
from services import training_service

# Temporary type-alias
# Evaluation = pd.DataFrame  # Dict[str, Any]
# Forecast = pd.DataFrame  # Dict[str, Any]


@dataclasses.dataclass
class TrainingJobManagerRequest:
    training_method: str
    dataset: dataset.Dataset
    start_time: datetime
    parameters: Dict
    id: str = dataclasses.field(default_factory=utils.generate_uuid)


class TrainingJobManager(abc.ABC):
    @abc.abstractmethod
    def enqueue_job(self, request: TrainingJobManagerRequest) -> str:
        pass

    @abc.abstractmethod
    def list_pending_jobs(self) -> List[Dict[str, Any]]:
        # TODO: Add pagination
        pass

    @abc.abstractmethod
    def get_evaluation(self, job_id: str) -> Optional[pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_forecast(self, job_id: str) -> Optional[pd.DataFrame]:
        pass


class MemoryTrainingJobManager(TrainingJobManager):
    """
    A job manager to queue jobs and delegate jobs to workers.

    Primarily used for development and testing.
    However, may be used in production if session affinity (https://cloud.google.com/run/docs/configuring/session-affinity) is enabled.
    """

    def __init__(self, training_service: training_service.TrainingService) -> None:
        super().__init__()
        self._training_service = training_service
        self._thread_pool_executor = futures.ThreadPoolExecutor()
        self._pending_jobs: Dict[str, TrainingJobManagerRequest] = {}
        self._completed_jobs: Dict[str, training_result.TrainingResult] = {}
        self._evaluation_uri_map: Dict[str, str] = {}
        self._forecast_uri_map: Dict[str, str] = {}

    def _process_request(self, request: TrainingJobManagerRequest):
        training_result = self._training_service.run(
            training_method_name=request.training_method,
            start_time=request.start_time,
            dataset=request.dataset,
            parameters=request.parameters,
        )

        return training_result

    def _append_completed_training_result(self, future: futures.Future):
        output: Tuple[str, training_result.TrainingResult] = future.result()

        # Deconstruct
        job_id, training_result = output

        if training_result:
            # Clear pending job
            del self._pending_jobs[job_id]

            # Append completed training results
            self._completed_jobs[job_id] = training_result

    def enqueue_job(self, request: TrainingJobManagerRequest) -> str:
        self._pending_jobs[request.id] = request

        future = self._thread_pool_executor.submit(self._process_request, request)
        future.add_done_callback(self._append_completed_training_result)

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
            for request in self._pending_jobs.values()
        ]

    def list_completed_jobs(self) -> List[training_result.TrainingResult]:
        # TODO: Add pagination
        return [
            {
                "start_time": result.start_time,
                "end_time": result.end_time,
                "error_message": result.error_message,
            }
            for result in self._completed_jobs.values()
        ]

    def _get_bigquery_table_as_df(self, table_id: str) -> pd.DataFrame:
        bigquery_uri = self._evaluation_uri_map.get(table_id)

        client = bigquery.Client()
        query = f"""
            SELECT *
            FROM `{bigquery_uri}`
        """

        query_job = client.query(
            query,
            # Location must match that of the dataset(s) referenced in the query.
            location="US",
        )  # API request - starts the query

        df = query_job.to_dataframe()
        return df

    def get_evaluation(self, job_id: str) -> Optional[pd.DataFrame]:
        job = self._completed_jobs.get(job_id)

        if job is None:
            return None

        table_id = job.evaluation_uri
        return self._get_bigquery_table_as_df(table_id=table_id) if table_id else None

    def get_forecast(self, job_id: str) -> Optional[pd.DataFrame]:
        job = self._completed_jobs.get(job_id)

        if job is None:
            return None

        table_id = job.forecast_uri
        return self._get_bigquery_table_as_df(table_id=table_id) if table_id else None
