from models import dataset
from typing import List

DATASETS = [
    dataset.CSVDataset(
        "sample_data/201306-citibike-tripdata.csv",
        display_name="NYC Bike Traffic",
        time_column="starttime",
        description="This dataset includes bike traffic data from CityBikes",
        icon="directions_bike",
    ),
    dataset.CSVDataset(
        "sample_data/sales_forecasting_full.csv",
        display_name="Retail Sales",
        time_column="date",
        description="This is sales data from a fictional sporting goods company with several stores across the city. It includes sales data for several products, grouped in several categories.",
        icon="storefront",
    ),
]


def get_datasets() -> List[dataset.Dataset]:
    return DATASETS
