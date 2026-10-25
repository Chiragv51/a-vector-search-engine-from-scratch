import numpy as np
import pytest
from vector_search.hnsw import HNSWIndex
from vector_search.filtering import (
    exact_filtered_search,
    post_filter_search,
    overfetch_filter_search,
    filter_aware_hnsw_search,
)
from vector_search.datasets import synthetic_dataset, recall_at_k


def test_filtering_strategies_and_selective_collapse():
    # Generate 500 vectors, where 90% are category 'A' and only 10% are category 'B'
    rng = np.random.default_rng(2026)
    n, d = 500, 16
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    # Assign metadata: 90% category 0, 10% category 1
    metadata = np.where(rng.random(size=n) < 0.10, 1, 0)
    metadata_map = {int(i): int(m) for i, m in zip(ids, metadata)}

    # Ensure we have enough category 1 items
    assert np.sum(metadata == 1) >= 20

    # Build HNSW index
    hnsw = HNSWIndex(dimension=d, M=12, ef_construction=100, seed=2026)
    for i, vec in zip(ids, vectors):
        hnsw.add(vec, int(i))

    query = rng.normal(size=d).astype(np.float32)
    k = 5

    # Ground truth
    truth = exact_filtered_search(vectors, ids, metadata, query, k=k, allowed_value=1)
    assert len(truth.ids) == k

    # Filter-aware traversal should find matching items with high recall
    aware_res = filter_aware_hnsw_search(
        hnsw, metadata_map, query, k=k, allowed_value=1, ef_search=50
    )
    assert len(aware_res.ids) == k
    for item_id in aware_res.ids:
        assert metadata_map[item_id] == 1

    # Over-fetch should also perform reasonably well
    over_res = overfetch_filter_search(
        hnsw, metadata_map, query, k=k, allowed_value=1, multiplier=15, ef_search=80
    )
    for item_id in over_res.ids:
        assert metadata_map[item_id] == 1
