import argparse
import statistics
import time

import requests


DEFAULT_PAYLOAD = {
    "VendorID": 1,
    "passenger_count": 1.0,
    "trip_distance": 2.5,
    "RatecodeID": 1.0,
    "PULocationID": 100,
    "DOLocationID": 110,
    "tpep_pickup_datetime": "2026-01-06T08:30:00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the online prediction API."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * percentile_value)
    return sorted_values[index]


def main() -> None:
    args = parse_args()

    for _ in range(args.warmup):
        response = requests.post(
            args.url,
            json=DEFAULT_PAYLOAD,
            timeout=10,
        )
        response.raise_for_status()

    latencies_ms: list[float] = []

    benchmark_start = time.perf_counter()

    for _ in range(args.requests):
        request_start = time.perf_counter()

        response = requests.post(
            args.url,
            json=DEFAULT_PAYLOAD,
            timeout=10,
        )
        response.raise_for_status()

        elapsed_ms = (time.perf_counter() - request_start) * 1000
        latencies_ms.append(elapsed_ms)

    total_seconds = time.perf_counter() - benchmark_start
    throughput = args.requests / total_seconds

    print("\n========== API BENCHMARK ==========")
    print(f"Requests   : {args.requests}")
    print(f"Mean       : {statistics.mean(latencies_ms):.2f} ms")
    print(f"p50        : {percentile(latencies_ms, 0.50):.2f} ms")
    print(f"p95        : {percentile(latencies_ms, 0.95):.2f} ms")
    print(f"Maximum    : {max(latencies_ms):.2f} ms")
    print(f"Throughput : {throughput:.2f} requests/second")


if __name__ == "__main__":
    main()