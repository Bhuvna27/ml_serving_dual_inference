from src.data.data_loader import load_dataset
from src.data.data_preprocessing import build_training_dataset
from src.data.data_validator import validate_dataset
from src.serving.predictor import predict


DATASET_FILENAME = "yellow_tripdata_2026-01.parquet"


def main():

    dataframe = load_dataset(DATASET_FILENAME)

    validate_dataset(dataframe)

    X, _, _ = build_training_dataset(dataframe)

    predictions = predict(X.head(5))

    print(predictions)


if __name__ == "__main__":
    main()