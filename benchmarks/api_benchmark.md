# Online API Benchmark

## Environment

- Machine: Apple MacBook Air
- Runtime: Python 3.11
- Server: Uvicorn, single local process
- Client and server: same machine
- Request mode: sequential
- Warm-up requests: 20
- Measured requests: 500

## Results

| Metric | Value |
|---|---:|
| Mean latency | 4.34 ms |
| p50 latency | 4.02 ms |
| p95 latency | 5.54 ms |
| Maximum latency | 45.77 ms |
| Throughput | 230.49 requests/second |
| Process RSS | 21.25 MB |

## Limitations

This was a local sequential benchmark, not a concurrent distributed load test. Results include HTTP serialization and localhost networking but do not represent production network latency or multi-client contention.