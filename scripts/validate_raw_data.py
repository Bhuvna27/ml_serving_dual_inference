"""
Command-line entry point for validating the raw NYC TLC dataset.

Run from the repository root:

    python -m scripts.validate_raw_data
"""

from src.data.data_loader import load_dataset
from src.data.data_validator import validate_dataset


DATASET_FILENAME = "yellow_tripdata_2026-01.parquet"


def main() -> None:
    """Load and validate the configured raw dataset."""

    dataframe = load_dataset(DATASET_FILENAME)

    validate_dataset(dataframe)

    print("Dataset validation passed.")
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")


if __name__ == "__main__":
    main()