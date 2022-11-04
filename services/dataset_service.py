import logging
from typing import List, Optional

from models import dataset

DATASETS = [
    dataset.CSVDataset(
        "sample_data/sales_forecasting_train.csv",
        display_name="Sport store sales (synthetic)",
        time_column="date",
        description="This is sales data from a fictional sporting goods company with several stores across the city. It includes sales data for several products, grouped in several categories.",
        icon="storefront",
        recommended_model_parameters={
            "bqml_arimaplus": {
                "targetColumn": "sales",
                "timeColumn": "date",
                "timeSeriesIdentifierColumn": "product_at_store",
                "dataFrequency": "daily",
            }
        },
        recommended_prediction_parameters={
            "bqml_arimaplus": {
                "forecastHorizon": 120,
            }
        },
    ),
    dataset.CSVDataset(
        "sample_data/iowa_liquor_sales.csv",
        display_name="Iowa Liquor Sales",
        time_column="date",
        description="This dataset contains the spirits purchase information of Iowa Class “E” liquor licensees by product and date of purchase. This dataset was simplified for demonstration purposes.",
        icon="liquor",
        recommended_model_parameters={
            "bqml_arimaplus": {
                "targetColumn": "sale_dollars",
                "timeColumn": "date",
                "timeSeriesIdentifierColumn": "county_and_city",
                "dataFrequency": "daily",
            }
        },
        recommended_prediction_parameters={
            "bqml_arimaplus": {
                "forecastHorizon": 120,
            }
        },
    ),
    # dataset.CSVDataset(
    #     "sample_data/cloud_spend.csv",
    #     display_name="Cloud Spend",
    #     time_column="date",
    #     description="This is exported Cloud Billing dataset from a sample GCP analytics use case project.",
    #     icon="cloud",
    #     recommended_model_parameters={
    #         "bqml_arimaplus": {
    #             "targetColumn": "cost",
    #             "timeColumn": "date",
    #             "timeSeriesIdentifierColumn": "project_name_and_service",
    #             "dataFrequency": "daily",
    #         }
    #     },
    #     recommended_prediction_parameters={
    #         "bqml_arimaplus": {
    #             "forecastHorizon": 120,
    #         }
    #     },
    # ),
]


def get_datasets() -> List[dataset.Dataset]:
    return DATASETS


def get_dataset(dataset_id: str) -> Optional[dataset.Dataset]:
    """Get the dataset given the dataset_id.

    Args:
        dataset_id (str): Dataset id.

    Returns:
        Optional[dataset.Dataset]: The dataset.
    """

    target_dataset = None

    for dataset in get_datasets():
        if str(dataset.id) == dataset_id:
            target_dataset = dataset
            break

    if target_dataset is not None:
        return target_dataset
    else:
        logging.error(f"Dataset id {dataset_id} does not exist!")
        return None
