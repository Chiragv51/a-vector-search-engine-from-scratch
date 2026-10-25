"""
Benchmark metadata filtering strategies across selectivities (100%, 50%, 10%, 1%, 0.1%).
Demonstrates the empirical collapse of naive post-filtering vs filter-aware traversal.
Saves results to benchmarks/results/filtering.json.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from vector_search.datasets import synthetic_dataset, recall_at_k
from vector_search.filtering import (
    exact_filtered_search,
    post_filter_search,
    overfetch_filter_search,
    filter_aware_hnsw_search,
)
from vector_search.hnsw import HNSWIndex


def main():
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    n, d, query_count = 5_000, 64, 50
    print(f"Generating filtering benchmark dataset (n={n:,}, d={d})...")
    rng = np.random.default_rng(2026)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    queries = rng.normal(size=(query_count, d)).astype(np.float32)

    # Build HNSW index once
    print("Building HNSW index...")
    hnsw = HNSWIndex(dimension=d, M=12, ef_construction=100, seed=2026)
    for i, vec in enumerate(vectors):
        hnsw.add(vec, i)

    # Test selectivities: 100%, 50%, 10%, 2%, 0.5%
    selectivities = [1.0, 0.5, 0.1, 0.02, 0.005]
    records = []
    k = 5

    print("\nEvaluating filtering strategies across selectivities...")
    for sel in selectivities:
        # Create metadata mask with probability ~ sel
        is_target = rng.random(size=n) < sel
        # Guarantee at least k target items
        if is_target.sum() < k:
            indices = rng.choice(n, size=k, replace=False)
            is_target[indices] = True

        metadata = np.where(is_target, 1, 0)
        metadata_map = {int(i): int(m) for i, m in zip(ids, metadata)}
        actual_pct = (metadata.sum() / n) * 100

        # Ground truth
        truth_list = []
        for q in queries:
            gt = exact_filtered_search(vectors, ids, metadata, q, k=k, allowed_value=1)
            truth_list.append(gt.ids)
        truth_ids = np.stack(truth_list)

        # 1. Post-filter strategy
        post_results = []
        t0 = time.perf_counter_ns()
        for q in queries:
            res = post_filter_search(hnsw, metadata_map, q, k=k, allowed_value=1, ef_search=50)
            # Pad with -1 if fewer than k returned
            padded = np.full(k, -1, dtype=np.int64)
            padded[:len(res.ids)] = res.ids[:k]
            post_results.append(padded)
        post_rec = recall_at_k(truth_ids, np.stack(post_results))

        # 2. Over-fetch strategy (10x multiplier)
        over_results = []
        for q in queries:
            res = overfetch_filter_search(
                hnsw, metadata_map, q, k=k, allowed_value=1, multiplier=10, ef_search=80
            )
            padded = np.full(k, -1, dtype=np.int64)
            padded[:len(res.ids)] = res.ids[:k]
            over_results.append(padded)
        over_rec = recall_at_k(truth_ids, np.stack(over_results))

        # 3. Filter-aware traversal
        aware_results = []
        for q in queries:
            res = filter_aware_hnsw_search(
                hnsw, metadata_map, q, k=k, allowed_value=1, ef_search=50
            )
            padded = np.full(k, -1, dtype=np.int64)
            padded[:len(res.ids)] = res.ids[:k]
            aware_results.append(padded)
        aware_rec = recall_at_k(truth_ids, np.stack(aware_results))

        records.append({
            "selectivity": sel,
            "actual_pct": actual_pct,
            "post_filter_recall": post_rec,
            "overfetch_recall": over_rec,
            "filter_aware_recall": aware_rec,
        })
        print(
            f"Selectivity={actual_pct:5.2f}% | Post-Filter={post_rec:.3f} | Over-Fetch={over_rec:.3f} | Filter-Aware={aware_rec:.3f}"
        )

    out_file = results_dir / "filtering.json"
    out_file.write_text(json.dumps({"results": records}, indent=2))
    print(f"\nFiltering results saved to {out_file}")


if __name__ == "__main__":
    main()
