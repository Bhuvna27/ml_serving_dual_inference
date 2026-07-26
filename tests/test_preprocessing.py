import pandas as pd

from src.config.ml_config import MODEL_FEATURE_COLUMNS
from src.data.data_preprocessing import build_training_dataset


def test_build_training_dataset_returns_expected_features() -> None:
    dataframe = pd.DataFrame(
        {
            "VendorID": [1, 2],
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-06 08:30:00", "2025-01-11 22:15:00"]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2025-01-06 08:45:00", "2025-01-11 22:40:00"]
            ),
            "passenger_count": [1.0, 2.0],
            "trip_distance": [2.5, 5.0],
            "RatecodeID": [1.0, 1.0],
            "PULocationID": [100, 120],
            "DOLocationID": [110, 130],
            "total_amount": [18.5, 32.0],
        }
    )

    features, target, pickup_times = build_training_dataset(dataframe)

    assert list(features.columns) == MODEL_FEATURE_COLUMNS
    assert len(features) == 2
    assert len(target) == 2
    assert len(pickup_times) == 2
    assert target.tolist() == [18.5, 32.0]
    assert features["pickup_hour"].tolist() == [8, 22]
    assert features["is_weekend"].tolist() == [0, 1]