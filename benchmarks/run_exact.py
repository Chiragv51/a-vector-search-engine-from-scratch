"""
Benchmark exact search scaling across dataset sizes (1K, 10K, 100K).
Measures build time, P50/P95/P99 latency, and QPS.
Saves results and environment metadata to benchmarks/results/exact.json.
"""

from __future__ import annotations
import json
from pathlib import Path
import platform
import time
import numpy as np
import psutil
from vector_search.datasets import synthetic_dataset
from vector_search.exact_numpy import ExactNumpyIndex, ExactNumpyLowMemory


def percentile_ms(samples_ns: list[int], p: float) -> float:
    return float(np.percentile(samples_ns, p) / 1_000_000)


def main():
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Running Exact Search Scaling Benchmark...")
    rows = []
    # Test 1K, 10K, 50K (can test 100K depending on time)
    sizes = [1_000, 10_000, 50_000]

    for n in sizes:
        print(f"-> Generating dataset for n={n:,}, d=128...")
        vectors, queries = synthetic_dataset(n, d=128, query_count=120, seed=2026)

        started = time.perf_counter_ns()
        index = ExactNumpyIndex(vectors)
        build_ms = (time.perf_counter_ns() - started) / 1_000_000

        # Warm-up queries (do not record)
        for q in queries[:20]:
            index.search(q, k=10)

        # Timed queries
        samples = []
        for q in queries[20:]:
            s = time.perf_counter_ns()
            index.search(q, k=10)
            samples.append(time.perf_counter_ns() - s)

        row = {
            "n": n,
            "d": 128,
            "k": 10,
            "queries": len(samples),
            "p50_ms": percentile_ms(samples, 50),
            "p95_ms": percentile_ms(samples, 95),
            "p99_ms": percentile_ms(samples, 99),
            "qps": 1e9 / float(np.mean(samples)),
            "build_ms": build_ms,
            "vector_bytes": int(vectors.nbytes),
        }
        rows.append(row)
        print(
            f"   n={n:,} | P50={row['p50_ms']:.2f}ms | P99={row['p99_ms']:.2f}ms | QPS={row['qps']:.1f}"
        )

    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total,
        },
        "results": rows,
    }

    output_path = results_dir / "exact.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
