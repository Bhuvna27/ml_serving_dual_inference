"""
Run batch predictions on the NYC Taxi dataset.
"""

from pathlib import Path

import pandas as pd

from src.data.data_loader import load_dataset
from src.data.data_preprocessing import build_training_dataset
from src.data.data_validator import validate_dataset
from src.serving.predictor import predict


DATASET_FILENAME = "yellow_tripdata_2026-01.parquet"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "predictions.csv"


def main():

    print("Loading dataset...")

    dataframe = load_dataset(DATASET_FILENAME)

    validate_dataset(dataframe)

    print("Preprocessing dataset...")

    X, _, _ = build_training_dataset(dataframe)

    print("Generating predictions...")

    predictions = predict(X)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dataframe = X.copy()

    prediction_dataframe["predicted_total_amount"] = predictions

    prediction_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nPredictions saved to:\n{OUTPUT_FILE}")

    print(f"\nTotal predictions: {len(prediction_dataframe):,}")


if __name__ == "__main__":
    main()