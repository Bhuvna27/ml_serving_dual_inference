from src.data.loader import load_dataset


def main():

    df = load_dataset(
        "yellow_tripdata_2026-01.parquet"
    )

    print(df.head())

    print()

    print(df.shape)


if __name__ == "__main__":
    main()