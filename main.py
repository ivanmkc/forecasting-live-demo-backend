import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException

from services import dataset_service, forecast_job_coordinator, forecast_job_service
from training_methods import bqml_training_method, training_method

logger = logging.getLogger(__name__)
from typing import Any, Dict, Optional

from pydantic import BaseModel

from services import dataset_service

app = FastAPI()

# TODO: Auto-detect registry
training_registry: Dict[str, training_method.TrainingMethod] = {
    bqml_training_method.BQMLARIMAPlusTrainingMethod.training_method(): bqml_training_method.BQMLARIMAPlusTrainingMethod()
}

training_service_instance = forecast_job_service.ForecastJobService(
    training_registry=training_registry
)
training_jobs_manager_instance = forecast_job_coordinator.MemoryTrainingJobManager(
    training_service=training_service_instance
)


@app.get("/datasets")
async def datasets():
    return [dataset.to_dict() for dataset in dataset_service.get_datasets()]


@app.get("/dataset/{dataset_id}")
def get_dataset(dataset_id: str):
    dataset = dataset_service.get_dataset(dataset_id=dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset id {dataset_id} was not found!"
        )
    else:
        return dataset


@app.get("/preview_dataset/{dataset_id}")
def preview_dataset(dataset_id: str):

    target_dataset = dataset_service.get_dataset(dataset_id)

    if target_dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset id {dataset_id} was not found!"
        )
    else:
        return target_dataset.df_preview.to_dict(orient="records")


@app.get("/pending_jobs")
def pending_jobs():
    return training_jobs_manager_instance.list_pending_jobs()


@app.get("/completed_jobs")
def completed_jobs():
    return training_jobs_manager_instance.list_completed_jobs()


class ForecastJobAPIRequest(BaseModel):
    """A forecast job request includes information to train a model, evaluate it and create a forecast prediction.

    Args:
        training_method (str): The unique key associated with a training method.
        dataset_id (str): The dataset id to be used for training.
        model_parameters (Dict[str, Any]): Parameters for training.
        prediction_parameters (Dict[str, Any]): Parameters for training.
    """

    training_method: str
    dataset_id: str
    model_parameters: Optional[Dict[str, Any]] = None
    prediction_parameters: Optional[Dict[str, Any]] = None


@app.post("/train")
def train(
    request: ForecastJobAPIRequest,
):
    dataset = dataset_service.get_dataset(dataset_id=request.dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {request.dataset_id}"
        )

    job_id = training_jobs_manager_instance.enqueue_job(
        forecast_job_coordinator.ForecastJobRequest(
            start_time=datetime.now(),
            training_method=request.training_method,
            dataset=dataset,
            model_parameters=request.model_parameters or {},
            prediction_parameters=request.prediction_parameters or {},
        )
    )

    return {"job_id": job_id}


# Get evaluation
@app.get("/evaluation/{job_id}")
async def evaluation(job_id: str):
    evaluation = training_jobs_manager_instance.get_evaluation(job_id=job_id)

    if evaluation is None:
        raise HTTPException(status_code=404, detail=f"Evaluation not found: {job_id}")
    else:
        return evaluation.to_json(orient="records")


# Get prediction
@app.get("/prediction/{job_id}")
async def prediction(job_id: str):
    prediction = training_jobs_manager_instance.get_prediction(job_id=job_id)

    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Prediction not found: {job_id}")
    else:
        return prediction.to_json(orient="records")
