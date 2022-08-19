from models import dataset
import json
from typing import List


def get_datasets() -> List[dataset.Dataset]:
    return [
        dataset.CSVDataset(
            "sample_data/201306-citibike-tripdata.csv",
            display_name="NYC Bike Traffic",
            time_column="starttime",
        ),
        dataset.CSVDataset(
            "sample_data/201306-citibike-tripdata.csv",
            display_name="Retail Sales",
            time_column="starttime",
        ),
    ]
