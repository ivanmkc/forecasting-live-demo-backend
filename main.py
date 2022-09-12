import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException

from services import dataset_service, training_jobs_manager, training_service
from training_methods import training_method, bqml_training_method

logger = logging.getLogger(__name__)
from typing import Any, Dict, Optional

from pydantic import BaseModel

from services import dataset_service

app = FastAPI()

# TODO: Auto-detect registry
training_registry: Dict[str, training_method.TrainingMethod] = {
    bqml_training_method.BQMLARIMAPlusTrainingMethod.training_method(): bqml_training_method.BQMLARIMAPlusTrainingMethod()
}

training_service_instance = training_service.TrainingService(
    training_registry=training_registry
)
training_jobs_manager_instance = training_jobs_manager.MemoryTrainingJobManager(
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


class TrainAPIRequest(BaseModel):
    training_method: str
    dataset_id: str
    model_parameters: Optional[Dict[str, Any]] = None
    forecast_parameters: Optional[Dict[str, Any]] = None


@app.post("/train")
def train(
    request: TrainAPIRequest,
):
    dataset = dataset_service.get_dataset(dataset_id=request.dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {request.dataset_id}"
        )

    job_id = training_jobs_manager_instance.enqueue_job(
        training_jobs_manager.TrainingJobManagerRequest(
            start_time=datetime.now(),
            training_method=request.training_method,
            dataset=dataset,
            model_parameters=request.model_parameters or {},
            forecast_parameters=request.forecast_parameters or {},
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


# Get forecast
@app.get("/forecast/{job_id}")
async def forecast(job_id: str):
    forecast = training_jobs_manager_instance.get_forecast(job_id=job_id)

    if forecast is None:
        raise HTTPException(status_code=404, detail=f"Forecast not found: {job_id}")
    else:
        return forecast.to_json(orient="records")


class ForecastRequest(BaseModel):
    model_id: str
    forecast_horizon: int


# @app.post("/forecast")
# def forecast(request: ForecastRequest):
# dataset = dataset_service.get_dataset(dataset_id=request.dataset_id)

# if dataset is None:
#     raise HTTPException(
#         status_code=404, detail=f"Dataset not found: {request.dataset_id}"
#     )

# job_id = training_jobs_manager_instance.enqueue(
#     training_jobs_manager.TrainingJobManagerRequest(
#         start_time=datetime.now(),
#         training_method=request.type,
#         dataset=dataset,
#         parameters={
#             "time_column": request.time_column,
#             "target_column": request.target_column,
#             "time_series_id_column": request.time_series_id_column,
#         },
#     )
# )

# return {"job_id": job_id}


# Vertex AI forecasting

# TODO: Get historical forecasts (and pending jobs)
