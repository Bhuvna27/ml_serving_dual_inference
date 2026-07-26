import argparse
from pathlib import Path

import pandas as pd

from src.data.data_loader import load_dataset
from src.data.data_preprocessing import build_training_dataset
from src.data.data_validator import validate_dataset
from src.serving.predictor import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch fare predictions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input Parquet filename located in data/raw.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input)
    validate_dataset(df)

    features, _, _ = build_training_dataset(df)
    predictions = predict(features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scored_df = df.loc[features.index].copy()
    scored_df["predicted_total_amount"] = predictions
    scored_df.to_csv(output_path, index=False)

    print(f"Saved {len(scored_df):,} predictions to {output_path}")


if __name__ == "__main__":
    main()