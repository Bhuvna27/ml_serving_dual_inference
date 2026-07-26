from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_with_valid_input() -> None:
    payload = {
        "VendorID": 1,
        "passenger_count": 1.0,
        "trip_distance": 2.5,
        "RatecodeID": 1.0,
        "PULocationID": 100,
        "DOLocationID": 110,
        "tpep_pickup_datetime": "2025-01-06T08:30:00",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_total_amount" in response.json()
    assert isinstance(response.json()["predicted_total_amount"], float)


def test_predict_endpoint_rejects_invalid_input() -> None:
    payload = {
        "VendorID": 1,
        "passenger_count": 1.0,
        "trip_distance": -2.5,
        "RatecodeID": 1.0,
        "PULocationID": 100,
        "DOLocationID": 110,
        "tpep_pickup_datetime": "2025-01-06T08:30:00",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422