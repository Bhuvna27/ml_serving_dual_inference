"""
FastAPI application for fare prediction.
"""

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    FarePredictionRequest,
    FarePredictionResponse,
)
from src.serving.predictor import predict


app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "NYC Taxi Fare Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=FarePredictionResponse,
)
def predict_fare(request: FarePredictionRequest):

    try:
        input_dataframe = pd.DataFrame(
            [request.model_dump()]
        )

        prediction = predict(input_dataframe)

        return FarePredictionResponse(
            predicted_total_amount=round(
                float(prediction[0]),
                2,
            )
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from error