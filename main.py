import logging
from datetime import datetime
import pandas as pd

from fastapi import FastAPI, HTTPException

from services import dataset_service, forecast_job_coordinator, forecast_job_service
from training_methods import (
    bqml_training_method,
    debug_training_method,
    training_method,
)

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from models import forecast_job_request
from services import dataset_service

app = FastAPI()

# TODO: Auto-detect registry
training_registry: Dict[str, training_method.TrainingMethod] = {
    bqml_training_method.BQMLARIMAPlusTrainingMethod.training_method(): bqml_training_method.BQMLARIMAPlusTrainingMethod(),
    debug_training_method.DebugTrainingMethod.training_method(): debug_training_method.DebugTrainingMethod(),
}

training_service_instance = forecast_job_service.ForecastJobService(
    training_registry=training_registry
)
training_jobs_manager_instance = forecast_job_coordinator.MemoryTrainingJobManager(
    forecast_job_service=training_service_instance
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
    jobs = training_jobs_manager_instance.list_pending_jobs()
    return [
        {
            "job_id": request.id,
            "training_method_name": request.training_method_name,
            "dataset": {
                "id": request.dataset.id,
                "icon": request.dataset.icon,
                "display_name": request.dataset.display_name,
            },
            "model_parameters": request.model_parameters,
            "prediction_parameters": request.prediction_parameters,
            "start_time": request.start_time,
        }
        for request in jobs
    ]


@app.get("/completed_jobs")
def completed_jobs():
    jobs = training_jobs_manager_instance.list_completed_jobs()
    return [
        {
            "job_id": job.request.id,
            "request": {
                "training_method_name": job.request.training_method_name,
                "dataset": {
                    "id": job.request.dataset.id,
                    "icon": job.request.dataset.icon,
                    "display_name": job.request.dataset.display_name,
                },
                "model_parameters": job.request.model_parameters,
                "prediction_parameters": job.request.prediction_parameters,
                "start_time": job.request.start_time,
            },
            "end_time": job.end_time,
            "error_message": job.error_message,
        }
        for job in jobs
    ]


class ForecastJobAPIRequest(BaseModel):
    """A forecast job request includes information to train a model, evaluate it and create a forecast prediction.

    Args:
        training_method_name (str): The unique key associated with a training method.
        dataset_id (str): The dataset id to be used for training.
        model_parameters (Dict[str, Any]): Parameters for training.
        prediction_parameters (Dict[str, Any]): Parameters for training.
    """

    training_method_name: str
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

    try:
        job_id = training_jobs_manager_instance.enqueue_job(
            forecast_job_request.ForecastJobRequest(
                start_time=datetime.now(),
                training_method_name=request.training_method_name,
                dataset=dataset,
                model_parameters=request.model_parameters or {},
                prediction_parameters=request.prediction_parameters or {},
            )
        )
    except Exception as exception:
        logger.error(str(exception))
        raise HTTPException(
            status_code=400, detail=f"There was a problem enqueueing your job"
        )

    return {"job_id": job_id}


# Get evaluation
@app.get("/evaluation/{job_id}")
async def evaluation(job_id: str):
    evaluation = training_jobs_manager_instance.get_evaluation(job_id=job_id)

    if evaluation is None:
        raise HTTPException(status_code=404, detail=f"Evaluation not found: {job_id}")
    else:
        evaluation = evaluation.fillna("")
        evaluation["id"] = evaluation.index

        return {
            "columns": evaluation.columns.tolist(),
            "rows": evaluation.to_dict(orient="records"),
        }


def format_for_rechart(
    group_column: str, time_column: str, target_column: str, data: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], Optional[datetime], Optional[datetime]]:

    data_grouped = data.groupby(group_column)

    # Creeate a map of group to map of time-to-values
    # i.e. group_time_value_map[group_id][time_id] = target_value
    group_time_value_map = {
        k: dict(zip(v[time_column].tolist(), v[target_column].tolist()))
        for k, v in data_grouped
    }

    unique_times = sorted(list(set(data[time_column].tolist())))

    data = [
        {
            "name": time,
            **{
                group: time_values_map.get(time)
                for group, time_values_map in group_time_value_map.items()
            },
        }
        for time in unique_times
    ]

    return (
        data,
        unique_times[0] if len(unique_times) > 0 else None,
        unique_times[-1] if len(unique_times) > 0 else None,
    )


# Get prediction
@app.get("/prediction/{job_id}/{output_type}")
async def prediction(job_id: str, output_type: str):
    try:
        job_request = training_jobs_manager_instance.get_request(job_id=job_id)

        if job_request is None:
            raise HTTPException(
                status_code=404, detail=f"Job request not found: {job_id}"
            )

        df_history = job_request.dataset.df
        df_prediction = training_jobs_manager_instance.get_prediction(job_id=job_id)
    except Exception as exception:
        logger.error(str(exception))
        raise HTTPException(
            status_code=400, detail=f"There was a problem getting prediction: {job_id}"
        )

    if df_prediction is None:
        raise HTTPException(status_code=404, detail=f"Prediction not found: {job_id}")
    else:
        df_prediction = df_prediction.fillna("")
        if output_type == "datagrid":
            df_prediction["id"] = df_prediction.index

            return {
                "columns": df_prediction.columns.tolist(),
                "rows": df_prediction.to_dict(orient="records"),
            }
        elif output_type == "chartjs":
            group_column = "product_at_store"  # Figure out how to generalize this.
            time_column = "forecast_timestamp"
            target_column = "forecast_value"  # Figure out how to generalize this.

            prediction_grouped = df_prediction.groupby(group_column)

            group_time_value_map = {
                k: dict(zip(v[time_column].tolist(), v[target_column].tolist()))
                for k, v in prediction_grouped
            }

            unique_times = sorted(list(df_prediction[time_column].unique()))

            datasets = [
                {
                    "label": group,
                    "data": [time_values_map[time] for time in unique_times],
                }
                for group, time_values_map in group_time_value_map.items()
            ]

            return {
                "time_labels": unique_times,
                "datasets": datasets,
            }
        elif output_type == "recharts":
            group_column = "product_at_store"  # Figure out how to generalize this.
            time_column = "forecast_timestamp"
            target_column = "forecast_value"  # Figure out how to generalize this.

            history_formatted, history_min_date, history_max_date = format_for_rechart(
                group_column=job_request.model_parameters["time_series_id_column"],
                time_column=job_request.model_parameters["time_column"],
                target_column=job_request.model_parameters["target_column"],
                data=df_history,
            )

            (
                predictions_formatted,
                predictions_min_date,
                predictions_max_date,
            ) = format_for_rechart(
                group_column=group_column,
                time_column=time_column,
                target_column=target_column,
                data=df_prediction,
            )

            return {
                "groups": df_prediction[group_column].unique().tolist(),
                "data": history_formatted + predictions_formatted,
                # "history": history_formatted,
                "history_min_date": history_min_date,
                "history_max_date": history_max_date,
                # "predictions": predictions_formatted,
                "predictions_min_date": predictions_min_date,
                "predictions_max_date": predictions_max_date,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output type: {output_type}",
            )
