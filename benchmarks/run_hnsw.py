"""
Benchmark HNSW parameter sweeps and recall vs latency trade-offs.
Generates frozen ground truth on exact search and sweeps efSearch.
Saves results to benchmarks/results/hnsw.json.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import psutil
from vector_search.datasets import synthetic_dataset, recall_at_k
from vector_search.exact_numpy import ExactNumpyIndex
from vector_search.hnsw import HNSWIndex


def percentile_ms(samples_ns: list[int], p: float) -> float:
    return float(np.percentile(samples_ns, p) / 1_000_000)


def main():
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    n, d, query_count = 5_000, 128, 100
    print(f"Generating benchmark dataset (n={n:,}, d={d}, queries={query_count})...")
    vectors, queries = synthetic_dataset(n, d, query_count=query_count, seed=2026)

    # 1. Compute & Freeze Ground Truth
    gt_file = results_dir / "ground_truth.npz"
    print("Computing exact ground truth with ExactNumpyIndex...")
    exact_idx = ExactNumpyIndex(vectors)
    truth_ids = np.stack([exact_idx.search(q, k=10).ids for q in queries])
    np.savez_compressed(
        gt_file,
        queries=queries,
        top10=truth_ids,
        seed=2026,
        metric="squared_l2",
    )
    print(f"Ground truth saved to {gt_file}")

    # 2. HNSW Parameter Configurations
    configs = [
        {"M": 8, "ef_construction": 64},
        {"M": 12, "ef_construction": 100},
        {"M": 16, "ef_construction": 128},
    ]
    ef_search_vals = [10, 20, 40, 80, 120]

    all_records = []

    for cfg in configs:
        M = cfg["M"]
        efC = cfg["ef_construction"]
        print(f"\nBuilding HNSW Index (M={M}, efConstruction={efC})...")
        proc = psutil.Process()
        mem_before = proc.memory_info().rss

        start_build = time.perf_counter_ns()
        index = HNSWIndex(dimension=d, M=M, ef_construction=efC, seed=2026)
        for i, vec in enumerate(vectors):
            index.add(vec, i)
        build_sec = (time.perf_counter_ns() - start_build) / 1e9
        mem_after = proc.memory_info().rss
        delta_mb = (mem_after - mem_before) / (1024 * 1024)
        print(f"-> Build complete in {build_sec:.2f}s (estimated graph memory ~{delta_mb:.1f} MB)")

        for efS in ef_search_vals:
            # Warm-up
            for q in queries[:10]:
                index.search(q, k=10, ef_search=efS)

            samples = []
            approx_results = []
            for q in queries:
                s = time.perf_counter_ns()
                res = index.search(q, k=10, ef_search=efS)
                samples.append(time.perf_counter_ns() - s)
                approx_results.append(res.ids)

            approx_ids = np.stack(approx_results)
            recall = recall_at_k(truth_ids, approx_ids)
            p50 = percentile_ms(samples, 50)
            p95 = percentile_ms(samples, 95)
            p99 = percentile_ms(samples, 99)
            qps = 1e9 / float(np.mean(samples))

            rec = {
                "engine": "hnsw_scratch",
                "n": n,
                "d": d,
                "M": M,
                "ef_construction": efC,
                "ef_search": efS,
                "recall_at_10": recall,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "qps": qps,
                "build_s": build_sec,
                "graph_mb": delta_mb,
            }
            all_records.append(rec)
            print(
                f"   efSearch={efS:3d} | Recall@10={recall:.4f} | P50={p50:.2f}ms | P99={p99:.2f}ms | QPS={qps:.0f}"
            )

    output_path = results_dir / "hnsw.json"
    output_path.write_text(json.dumps({"results": all_records}, indent=2))
    print(f"\nHNSW benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()
