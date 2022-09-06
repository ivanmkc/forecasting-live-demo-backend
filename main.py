import logging

from fastapi import FastAPI, HTTPException

from services import dataset_service, training_service

logger = logging.getLogger(__name__)
from pydantic import BaseModel

from services import dataset_service

app = FastAPI()


@app.get("/get_datasets")
async def get_datasets():
    return [dataset.to_dict() for dataset in dataset_service.get_datasets()]


@app.get("/get_dataset/{dataset_id}")
def get_dataset(dataset_id: str):
    dataset = dataset_service.get_dataset(dataset_id=dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset id {dataset_id} was not found!"
        )
    else:
        return dataset


@app.get("/dataset_data/{dataset_id}")
def get_dataset_data(dataset_id: str):
    dataset = dataset_service.get_dataset(dataset_id=dataset_id)

    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
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


# TODO: Move this
# BQML ARIMA+
training_registry = {"bqml": training_service.train_bigquery}


class TrainRequest(BaseModel):
    type: str
    dataset_id: str
    time_column: str
    target_column: str
    time_series_id_column: str


@app.post("/train")
def train(request: TrainRequest):
    train_method = training_registry.get(request.type)

    dataset = dataset_service.get_dataset(dataset_id=request.dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {request.dataset_id}"
        )

    if train_method:
        try:
            forecast = train_method(
                dataset=dataset,
                time_column=request.time_column,
                target_column=request.target_column,
                time_series_id_column=request.time_series_id_column,
            )
        except Exception as exception:
            raise HTTPException(status_code=500, detail=str(exception))
    else:
        raise HTTPException(
            status_code=404, detail=f"Training method not supported: {request.type}"
        )


# Vertex AI forecasting

# TODO: Get historical forecasts (and pending jobs)
