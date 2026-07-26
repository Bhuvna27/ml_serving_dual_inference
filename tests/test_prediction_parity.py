import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.config.ml_config import MODEL_FEATURE_COLUMNS
from src.serving.predictor import predict


client = TestClient(app)


def test_api_and_shared_predictor_return_same_prediction() -> None:
    api_payload = {
        "VendorID": 1,
        "passenger_count": 1.0,
        "trip_distance": 2.5,
        "RatecodeID": 1.0,
        "PULocationID": 100,
        "DOLocationID": 110,
        "tpep_pickup_datetime": "2025-01-06T08:30:00",
    }

    pickup_datetime = pd.Timestamp(api_payload["tpep_pickup_datetime"])

    model_payload = {
        "VendorID": api_payload["VendorID"],
        "passenger_count": api_payload["passenger_count"],
        "trip_distance": api_payload["trip_distance"],
        "RatecodeID": api_payload["RatecodeID"],
        "PULocationID": api_payload["PULocationID"],
        "DOLocationID": api_payload["DOLocationID"],
        "pickup_hour": pickup_datetime.hour,
        "pickup_day_of_week": pickup_datetime.dayofweek,
        "pickup_month": pickup_datetime.month,
        "is_weekend": int(pickup_datetime.dayofweek >= 5),
    }

    features = pd.DataFrame(
        [model_payload],
        columns=MODEL_FEATURE_COLUMNS,
    )

    shared_prediction = float(predict(features)[0])

    response = client.post("/predict", json=api_payload)

    assert response.status_code == 200

    api_prediction = response.json()["predicted_total_amount"]

    assert np.isclose(
        api_prediction,
        round(shared_prediction, 2),
        rtol=1e-7,
        atol=1e-7,
    )