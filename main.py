from queue import Empty
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
def get_dataset(dataset_id: str):
  target_dataset = dataset_service.get_dataset(dataset_id)
  return target_dataset

@app.get("/preview_dataset/{dataset_id}")
def preview_dataset(dataset_id: str):

    target_dataset = dataset_service.get_dataset(dataset_id)

    if target_dataset is not None:
      return target_dataset.df_preview.to_dict("records")
    else:
      return None

@app.get("/dataset_data/{dataset_id}")
def get_dataset_data(dataset_id: str):
  target_dataset = dataset_service.get_dataset(dataset_id)

  if target_dataset is not None:
    return target_dataset.df.to_json()
  else:
    return None


# TODO: Train a model
# BQML ARIMA+
# Vertex AI forecasting

# TODO: Get historical forecasts (and pending jobs)
