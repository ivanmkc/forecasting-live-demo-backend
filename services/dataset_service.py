from typing import List, Optional

from models import dataset

import logging
from typing import List, Optional

DATASETS = [
    dataset.CSVDataset(
        "sample_data/sales_forecasting_train.csv",
        display_name="Retail Sales",
        time_column="date",
        description="This is sales data from a fictional sporting goods company with several stores across the city. It includes sales data for several products, grouped in several categories.",
        icon="storefront",
    ),
    dataset.CSVDataset(
        "sample_data/201306-citibike-tripdata.csv",
        display_name="NYC Bike Traffic",
        time_column="starttime",
        description="This dataset includes bike traffic data from CityBikes",
        icon="directions_bike",
    ),
]


def get_datasets() -> List[dataset.Dataset]:
    return DATASETS


def get_dataset(dataset_id: str) -> Optional[dataset.Dataset]:
    """This functions returns a dataset based on the dataset id as input."""

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
