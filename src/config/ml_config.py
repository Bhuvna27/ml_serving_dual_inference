"""
Machine learning configuration for total fare prediction.
"""

TARGET_COLUMN = "total_amount"

FEATURE_COLUMNS = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]

LEAKAGE_COLUMNS = [
    "fare_amount",
    "tip_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]