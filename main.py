from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from starlette.background import BackgroundTask

from pathlib import Path
from typing import Union

from services import dataset_service

import logging

logger = logging.getLogger(__name__)

import os
import uuid
from services import dataset_service

app = FastAPI()


@app.get("/get_datasets")
async def get_datasets():
    return [dataset.to_dict() for dataset in dataset_service.get_datasets()]


@app.get("/get_dataset/{dataset_id}")
def get_dataset(dataset_id: int):
    return {}


# TODO: Train a model
# BQML ARIMA+
# Vertex AI forecasting

# TODO: Get historical forecasts (and pending jobs)
