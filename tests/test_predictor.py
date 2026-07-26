import numpy as np
import pandas as pd
import pytest

from src.config.ml_config import MODEL_FEATURE_COLUMNS
from src.serving.predictor import predict


@pytest.fixture
def valid_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "VendorID": 1,
                "passenger_count": 1.0,
                "trip_distance": 2.5,
                "RatecodeID": 1.0,
                "PULocationID": 100,
                "DOLocationID": 110,
                "pickup_hour": 8,
                "pickup_day_of_week": 0,
                "pickup_month": 1,
                "is_weekend": 0,
            },
            {
                "VendorID": 2,
                "passenger_count": 2.0,
                "trip_distance": 5.0,
                "RatecodeID": 1.0,
                "PULocationID": 120,
                "DOLocationID": 130,
                "pickup_hour": 22,
                "pickup_day_of_week": 5,
                "pickup_month": 1,
                "is_weekend": 1,
            },
        ],
        columns=MODEL_FEATURE_COLUMNS,
    )


def test_predict_returns_one_prediction_per_row(
    valid_features: pd.DataFrame,
) -> None:
    predictions = predict(valid_features)

    assert len(predictions) == len(valid_features)
    assert np.isfinite(predictions).all()


def test_predict_rejects_missing_feature(
    valid_features: pd.DataFrame,
) -> None:
    invalid_features = valid_features.drop(columns=["trip_distance"])

    with pytest.raises((ValueError, KeyError)):
        predict(invalid_features)