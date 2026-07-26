"""
Train a baseline model for total fare prediction.
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config.ml_config import RANDOM_SEED

from src.data.data_loader import load_dataset
from src.data.data_preprocessing import build_training_dataset
from src.data.data_validator import validate_dataset


DATASET_FILENAME = "yellow_tripdata_2026-01.parquet"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"


def main():

    print("Loading dataset...")

    dataframe = load_dataset(DATASET_FILENAME)

    validate_dataset(dataframe)

    print("Preprocessing dataset...")

    X, y, pickup_times = build_training_dataset(dataframe)

    print("Sorting dataset by pickup time...")

    sorted_index = pickup_times.sort_values().index

    X = X.loc[sorted_index]
    y = y.loc[sorted_index]

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows : {len(X_train):,}")
    print(f"Testing rows  : {len(X_test):,}")

    print("Training model...")

    model = HistGradientBoostingRegressor(
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\n========== MODEL METRICS ==========")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "fare_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    print(f"\nModel saved to:\n{model_path}")


    metrics = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    }

    metrics_path = PROJECT_ROOT / "models" / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved metrics to {metrics_path}")

    metadata = {
        "model_type": type(model).__name__,
        "target": "total_amount",
        "feature_columns": list(X_train.columns),
        "random_seed": RANDOM_SEED,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "NYC TLC Yellow Taxi Trip Records",
        "dataset_period": "2026-01",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics_file": "metrics.json",
    }

    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved model metadata to {metadata_path}")


if __name__ == "__main__":
    main()