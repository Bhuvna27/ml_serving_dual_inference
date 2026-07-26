"""
Request and response schemas for the fare prediction API.
"""

from pydantic import BaseModel, Field


class FarePredictionRequest(BaseModel):
    VendorID: int = Field(ge=1)
    passenger_count: float = Field(ge=0, le=9)
    trip_distance: float = Field(gt=0)
    RatecodeID: float = Field(ge=1)
    PULocationID: int = Field(ge=1)
    DOLocationID: int = Field(ge=1)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_day_of_week: int = Field(ge=0, le=6)
    pickup_month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)


class FarePredictionResponse(BaseModel):
    predicted_total_amount: float