from datetime import datetime

from pydantic import BaseModel, Field


class FarePredictionRequest(BaseModel):
    VendorID: int = Field(..., ge=1)
    passenger_count: float = Field(..., ge=0, le=9)
    trip_distance: float = Field(..., gt=0)
    RatecodeID: float = Field(..., ge=1)
    PULocationID: int = Field(..., ge=1)
    DOLocationID: int = Field(..., ge=1)
    tpep_pickup_datetime: datetime


class FarePredictionResponse(BaseModel):
    predicted_total_amount: float