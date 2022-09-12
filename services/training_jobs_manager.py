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


class ForecastJobManager(abc.ABC):
    """
    Manages the queue of jobs, listing pending jobs and getting results.
    A forecast job is defined as a pipeline involved training a model, getting evaluations and getting a prediction.
    """

    @abc.abstractmethod
    def enqueue_job(self, request: ForecastJobRequest) -> str:
        """Enqueue the request to a job queue for later processing.

        Args:
            request (TrainingJobManagerRequest): The job request.

        Returns:
            str: The job id
        """
        pass

    @abc.abstractmethod
    def list_pending_jobs(self) -> List[Dict[str, Any]]:
        """List pending jobs.

        Returns:
            List[Dict[str, Any]]: The pending jobs.
        """
        # TODO: Add pagination
        pass

    def list_completed_jobs(self) -> List[training_result.ForecastJobResult]:
        """List completed jobs.

        Returns:
            List[training_result.TrainingResult]: The completed results.
        """
        pass

    @abc.abstractmethod
    def get_evaluation(self, job_id: str) -> Optional[pd.DataFrame]:
        """Get the evaluation dataframe for a given job_id.

        Args:
            job_id (str): Job id.

        Returns:
            Optional[pd.DataFrame]: The evaluation dataframe.
        """
        pass

    @abc.abstractmethod
    def get_prediction(self, job_id: str) -> Optional[pd.DataFrame]:
        """Get the prediction dataframe for a given job_id.

        Args:
            job_id (str): Job id.

        Returns:
            Optional[pd.DataFrame]: The prediction dataframe.
        """
        pass


class MemoryTrainingJobManager(ForecastJobManager):
    """
    A job manager to queue jobs and delegate jobs to workers.

    Primarily used for development and testing.
    However, may be used in production if session affinity (https://cloud.google.com/run/docs/configuring/session-affinity) is enabled.
    """

    def __init__(self, training_service: training_service.TrainingJobService) -> None:
        """Initializes the manager.

        Args:
            training_service (training_service.TrainingJobService): The service used by each worker to run the training job.
        """
        super().__init__()
        self._training_service = training_service
        self._thread_pool_executor = futures.ThreadPoolExecutor()
        self._pending_jobs: Dict[str, ForecastJobRequest] = {}
        self._completed_jobs: Dict[str, training_result.ForecastJobResult] = {}
        self._evaluation_uri_map: Dict[str, str] = {}
        self._prediction_uri_map: Dict[str, str] = {}

    def _process_request(
        self, request: ForecastJobRequest
    ) -> Tuple[str, training_result.ForecastJobResult]:
        """Process the training jobs request.

        Args:
            request (TrainingJobManagerRequest): The request

        Returns:
            Tuple[str, training_result.TrainingResult]: The job id and result.
        """
        training_result = self._training_service.run(
            training_method_name=request.training_method,
            start_time=request.start_time,
            dataset=request.dataset,
            model_parameters=request.model_parameters,
            prediction_parameters=request.prediction_parameters,
        )

        return request.id, training_result

    def _append_completed_training_result(self, future: futures.Future):
        """Append the result to a cache. Used as a Future callback.

        Args:
            future (futures.Future): The future that will return the TrainingResult.
        """
        output: Tuple[str, training_result.TrainingResult] = future.result()

        # Deconstruct
        job_id, training_result = output

        if training_result:
            # Clear pending job
            del self._pending_jobs[job_id]

            # Append completed training results
            self._completed_jobs[job_id] = training_result

    def enqueue_job(self, request: ForecastJobRequest) -> str:
        """Enqueue the request to a job queue for later processing.

        Args:
            request (TrainingJobManagerRequest): The job request.

        Returns:
            str: The job id
        """
        self._pending_jobs[request.id] = request

        future = self._thread_pool_executor.submit(self._process_request, request)
        future.add_done_callback(self._append_completed_training_result)

        return request.id

    def list_pending_jobs(self) -> List[Dict[str, Any]]:
        """List pending jobs.

        Returns:
            List[Dict[str, Any]]: The pending jobs.
        """
        # TODO: Add pagination
        return [
            {
                "job_id": request.id,
                "training_method": request.training_method,
                "dataset_id": request.dataset.id,
                "model_parameters": request.model_parameters,
                "prediction_parameters": request.prediction_parameters,
                "start_time": request.start_time,
            }
            for request in self._pending_jobs.values()
        ]

    def list_completed_jobs(self) -> List[training_result.ForecastJobResult]:
        """List completed jobs.

        Returns:
            List[training_result.TrainingResult]: The completed results.
        """
        # TODO: Add pagination
        return [
            {
                "job_id": job_id,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "error_message": result.error_message,
            }
            for job_id, result in self._completed_jobs.items()
        ]

    def _get_bigquery_table_as_df(self, table_id: str) -> pd.DataFrame:
        client = bigquery.Client()
        query = f"""
            SELECT *
            FROM `{table_id}`
        """

        query_job = client.query(
            query=query,
        )

        df = query_job.to_dataframe()
        return df

    def get_evaluation(self, job_id: str) -> Optional[pd.DataFrame]:
        """Get the evaluation dataframe for a given job_id.

        Args:
            job_id (str): Job id.

        Returns:
            Optional[pd.DataFrame]: The evaluation dataframe.
        """
        job = self._completed_jobs.get(job_id)

        if job is None:
            return None

        table_id = job.evaluation_uri
        return self._get_bigquery_table_as_df(table_id=table_id) if table_id else None

    def get_prediction(self, job_id: str) -> Optional[pd.DataFrame]:
        """Get the prediction dataframe for a given job_id.

        Args:
            job_id (str): Job id.

        Returns:
            Optional[pd.DataFrame]: The prediction dataframe.
        """
        job = self._completed_jobs.get(job_id)

        if job is None:
            return None

        table_id = job.prediction_uri
        return self._get_bigquery_table_as_df(table_id=table_id) if table_id else None
