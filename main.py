import logging

from fastapi import FastAPI, HTTPException

from services import dataset_service, training_service
from training_methods import training_method
from services import dataset_service, training_jobs_manager, training_service

logger = logging.getLogger(__name__)
from pydantic import BaseModel

from services import dataset_service
from typing import Dict

app = FastAPI()

# TODO: Auto-detect registry
training_registry: Dict[str, training_method.TrainingMethod] = {
    "bqml": training_method.BQMLARIMAPlusTrainingMethod()
}

training_service_instance = training_service.TrainingService(
    training_registry=training_registry
)
training_jobs_manager_instance = training_jobs_manager.MemoryTrainingJobManager(
    training_service=training_service_instance
)


@app.get("/datasets")
async def get_datasets():
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
        return target_dataset.df_preview.to_dict("records")


class TrainRequest(BaseModel):
    type: str
    dataset_id: str
    time_column: str
    target_column: str
    time_series_id_column: str


@app.post("/train")
def train(request: TrainRequest):
    dataset = dataset_service.get_dataset(dataset_id=request.dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {request.dataset_id}"
        )

    train_method = training_registry.get(request.type)

    if train_method:
        # try:
        job_id = training_jobs_manager_instance.queue(
            dataset=dataset,
            time_column=request.time_column,
            target_column=request.target_column,
            time_series_id_column=request.time_series_id_column,
        )

        # TODO: Log job_id

        return job_id
        # except Exception as exception:
        #     raise FastAPIHTTPException(status_code=500, detail=str(exception))
    else:
        raise HTTPException(
            status_code=404, detail=f"Training method not supported: {request.type}"
        )


# Vertex AI forecasting

# TODO: Get historical forecasts (and pending jobs)
