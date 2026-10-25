"""
Fair head-to-head benchmark: Vector Search from Scratch vs FAISS.
Compares exact ground truth against faiss.IndexFlatL2 and evaluates
our HNSW against faiss.IndexHNSWFlat under identical conditions.
Saves results to benchmarks/results/comparison.json.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
import faiss
import numpy as np
from vector_search.datasets import synthetic_dataset, recall_at_k
from vector_search.exact_numpy import ExactNumpyIndex
from vector_search.hnsw import HNSWIndex


def percentile_ms(samples_ns: list[int], p: float) -> float:
    return float(np.percentile(samples_ns, p) / 1_000_000)


def main():
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    n, d, query_count = 5_000, 128, 100
    print(f"Generating fair comparison dataset (n={n:,}, d={d}, queries={query_count})...")
    vectors, queries = synthetic_dataset(n, d, query_count=query_count, seed=2026)

    # 1. Exact Reference Cross-Check
    print("\n--- Phase 1: Exact Reference Cross-Check ---")
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    q = np.ascontiguousarray(queries, dtype=np.float32)

    # Our exact index
    scratch_exact = ExactNumpyIndex(x)
    scratch_truth = np.stack([scratch_exact.search(query, 10).ids for query in q])

    # FAISS exact flat index
    faiss_flat = faiss.IndexFlatL2(d)
    faiss_flat.add(x)
    _, faiss_truth = faiss_flat.search(q, 10)

    # Verify identical top-k IDs (allowing deterministic tie tolerance)
    match_rate = float(np.mean(scratch_truth == faiss_truth))
    print(f"Exact match agreement between Scratch and FAISS IndexFlatL2: {match_rate * 100:.2f}%")

    # 2. HNSW Head-to-Head Comparison
    print("\n--- Phase 2: HNSW Head-to-Head Comparison ---")
    M = 12
    ef_c = 100
    ef_search_vals = [10, 20, 40, 80, 160]
    records = []

    # Build Scratch HNSW
    print("Building Scratch HNSW index...")
    t0 = time.perf_counter_ns()
    scratch_hnsw = HNSWIndex(dimension=d, M=M, ef_construction=ef_c, seed=2026)
    for i, vec in enumerate(x):
        scratch_hnsw.add(vec, i)
    scratch_build_s = (time.perf_counter_ns() - t0) / 1e9
    print(f"Scratch HNSW build time: {scratch_build_s:.2f}s")

    # Build FAISS HNSW
    print("Building FAISS IndexHNSWFlat...")
    t0 = time.perf_counter_ns()
    faiss_hnsw = faiss.IndexHNSWFlat(d, M)
    faiss_hnsw.hnsw.efConstruction = ef_c
    faiss_hnsw.add(x)
    faiss_build_s = (time.perf_counter_ns() - t0) / 1e9
    print(f"FAISS HNSW build time: {faiss_build_s:.2f}s")

    # Evaluate Scratch HNSW over ef_search sweep
    print("\nEvaluating Scratch HNSW...")
    for efS in ef_search_vals:
        # Warmup
        for query in q[:10]:
            scratch_hnsw.search(query, k=10, ef_search=efS)

        samples = []
        approx = []
        for query in q:
            s = time.perf_counter_ns()
            res = scratch_hnsw.search(query, k=10, ef_search=efS)
            samples.append(time.perf_counter_ns() - s)
            approx.append(res.ids)

        rec = recall_at_k(scratch_truth, np.stack(approx))
        p50 = percentile_ms(samples, 50)
        p95 = percentile_ms(samples, 95)
        p99 = percentile_ms(samples, 99)
        qps = 1e9 / float(np.mean(samples))
        records.append({
            "engine": "Scratch HNSW (Pure Python)",
            "ef_search": efS,
            "recall_at_10": rec,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "qps": qps,
            "build_s": scratch_build_s,
        })
        print(f"Scratch | efS={efS:3d} | Recall@10={rec:.4f} | P50={p50:.2f}ms | P99={p99:.2f}ms | QPS={qps:.0f}")

    # Evaluate FAISS HNSW over ef_search sweep (single-threaded for fair serial comparison)
    print("\nEvaluating FAISS HNSW (serial query evaluation)...")
    faiss.omp_set_num_threads(1)
    for efS in ef_search_vals:
        faiss_hnsw.hnsw.efSearch = efS
        # Warmup
        faiss_hnsw.search(q[:10], 10)

        samples = []
        approx = []
        for query in q:
            s = time.perf_counter_ns()
            _, ids = faiss_hnsw.search(query.reshape(1, -1), 10)
            samples.append(time.perf_counter_ns() - s)
            approx.append(ids[0])

        rec = recall_at_k(scratch_truth, np.stack(approx))
        p50 = percentile_ms(samples, 50)
        p95 = percentile_ms(samples, 95)
        p99 = percentile_ms(samples, 99)
        qps = 1e9 / float(np.mean(samples))
        records.append({
            "engine": "FAISS IndexHNSWFlat (C++)",
            "ef_search": efS,
            "recall_at_10": rec,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "qps": qps,
            "build_s": faiss_build_s,
        })
        print(f"FAISS   | efS={efS:3d} | Recall@10={rec:.4f} | P50={p50:.2f}ms | P99={p99:.2f}ms | QPS={qps:.0f}")

    output_path = results_dir / "comparison.json"
    output_path.write_text(json.dumps({"results": records}, indent=2))
    print(f"\nFair comparison benchmark saved to {output_path}")


if __name__ == "__main__":
    main()
