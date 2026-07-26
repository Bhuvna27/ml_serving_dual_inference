"""
Generate a basic profile of the raw NYC TLC dataset.
"""

from src.data.data_loader import load_dataset
from src.data.data_validator import validate_dataset
from src.data.data_profiler import profile_dataset


DATASET_FILENAME = "yellow_tripdata_2026-01.parquet"


def main():

    dataframe = load_dataset(DATASET_FILENAME)

    validate_dataset(dataframe)

    profile_dataset(dataframe)


if __name__ == "__main__":
    main()