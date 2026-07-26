"""
Smoke test for the preprocessing pipeline.

Run from the repository root:

    python -m scripts.test_preprocessing
"""

from src.data.data_loader import load_dataset
from src.data.data_preprocessing import build_training_dataset
from src.data.data_validator import validate_dataset


DATASET_FILENAME = "yellow_tripdata_2026-01.parquet"


def main() -> None:
    dataframe = load_dataset(DATASET_FILENAME)

    validate_dataset(dataframe)

    original_rows = len(dataframe)

    X, y = build_training_dataset(dataframe)

    removed_rows = original_rows - len(X)

    print("Preprocessing completed successfully.")
    print(f"Original rows: {original_rows:,}")
    print(f"Training rows: {len(X):,}")
    print(f"Removed rows: {removed_rows:,}")
    print(f"Feature count: {len(X.columns)}")

    print("\nModel features:")
    print(X.columns.tolist())

    print("\nFeature data types:")
    print(X.dtypes)

    print("\nRemaining null values:")
    print(X.isna().sum())

    print("\nTarget summary:")
    print(y.describe())


if __name__ == "__main__":
    main()