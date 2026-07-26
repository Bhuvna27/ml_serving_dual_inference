from fastapi import FastAPI, HTTPException
import pandas as pd

from src.api.schemas import FarePredictionRequest, FarePredictionResponse
from src.config.ml_config import MODEL_FEATURE_COLUMNS
from src.serving.predictor import predict


app = FastAPI(title="NYC Taxi Fare Prediction API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "NYC Taxi Fare Prediction API"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=FarePredictionResponse)
def predict_fare(request: FarePredictionRequest) -> FarePredictionResponse:
    try:
        pickup_datetime = pd.Timestamp(request.tpep_pickup_datetime)

        payload = {
            "VendorID": request.VendorID,
            "passenger_count": request.passenger_count,
            "trip_distance": request.trip_distance,
            "RatecodeID": request.RatecodeID,
            "PULocationID": request.PULocationID,
            "DOLocationID": request.DOLocationID,
            "pickup_hour": pickup_datetime.hour,
            "pickup_day_of_week": pickup_datetime.dayofweek,
            "pickup_month": pickup_datetime.month,
            "is_weekend": int(pickup_datetime.dayofweek >= 5),
        }

        features = pd.DataFrame(
            [payload],
            columns=MODEL_FEATURE_COLUMNS,
        )

        prediction = float(predict(features)[0])

        return FarePredictionResponse(
            predicted_total_amount=round(prediction, 2)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is unavailable.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from error

@app.get("/ready")
def readiness() -> dict[str, str]:
    """
    Readiness probe.
    Confirms that the model is loaded and ready to serve requests.
    """
    try:
        from src.serving.model_loader import load_model

        load_model()

        return {"status": "ready"}

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

@app.get("/model-info")
def model_info():
    import json
    from pathlib import Path

    metadata_path = Path("models/model_metadata.json")

    with metadata_path.open() as f:
        return json.load(f)