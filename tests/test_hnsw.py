import numpy as np
import pytest
from vector_search.exact_numpy import ExactNumpyIndex
from vector_search.hnsw import HNSWIndex
from vector_search.datasets import synthetic_dataset, recall_at_k


def assert_graph_invariants(index: HNSWIndex):
    """Verify structural HNSW graph invariants."""
    assert index.entry is not None
    assert index.levels[index.entry] == index.max_level

    for layer, nodes in index.links.items():
        max_deg = index.M0 if layer == 0 else index.M
        for node, neighbors in nodes.items():
            # Invariant 1: No self-loops
            assert node not in neighbors, f"Node {node} links to itself on layer {layer}"
            # Invariant 2: Degree limits
            assert len(neighbors) <= max_deg, (
                f"Node {node} degree {len(neighbors)} exceeds max {max_deg} on layer {layer}"
            )
            # Invariant 3: Node level matches
            assert index.levels[node] >= layer, (
                f"Node {node} maximum level {index.levels[node]} < layer {layer}"
            )
            # Invariant 4: Bidirectional edges & neighbor level matches
            for neighbor in neighbors:
                assert index.levels[neighbor] >= layer, (
                    f"Neighbor {neighbor} level {index.levels[neighbor]} < layer {layer}"
                )
                assert node in index.links[layer][neighbor], (
                    f"Edge between {node} and {neighbor} is not bidirectional on layer {layer}"
                )


def test_hnsw_graph_invariants_on_seeded_data():
    vectors, _ = synthetic_dataset(n=200, d=16, query_count=10, seed=42)
    index = HNSWIndex(dimension=16, M=8, ef_construction=64, seed=42)

    for item_id, vec in enumerate(vectors):
        index.add(vec, item_id)

    assert len(index) == 200
    assert_graph_invariants(index)


def test_hnsw_recall_monotonicity_with_ef_search():
    """As ef_search increases, recall@10 should increase or remain stable."""
    n, d, query_count = 500, 16, 50
    vectors, queries = synthetic_dataset(n=n, d=d, query_count=query_count, seed=2026)

    # Compute ground truth with ExactNumpyIndex
    exact_idx = ExactNumpyIndex(vectors)
    truth_ids = np.stack([exact_idx.search(q, k=10).ids for q in queries])

    # Build HNSW index
    hnsw = HNSWIndex(dimension=d, M=12, ef_construction=100, seed=2026)
    for item_id, vec in enumerate(vectors):
        hnsw.add(vec, item_id)

    recalls = []
    for ef in [10, 20, 40, 80]:
        approx_ids = np.stack([hnsw.search(q, k=10, ef_search=ef).ids for q in queries])
        rec = recall_at_k(truth_ids, approx_ids)
        recalls.append(rec)

    # High recall at ef=80
    assert recalls[-1] >= 0.85, f"Expected high recall at ef=80, got {recalls[-1]}"
    # Verify non-decreasing trend
    for i in range(len(recalls) - 1):
        assert recalls[i + 1] >= recalls[i] - 0.05, (
            f"Recall dropped significantly: {recalls[i]} -> {recalls[i+1]}"
        )


def test_hnsw_rejects_invalid_inputs():
    hnsw = HNSWIndex(dimension=4, M=8, ef_construction=32)
    hnsw.add([1, 2, 3, 4], 0)

    # Duplicate ID
    with pytest.raises(ValueError, match="duplicate item_id"):
        hnsw.add([1, 2, 3, 4], 0)

    # Wrong shape
    with pytest.raises(ValueError, match="invalid vector"):
        hnsw.add([1, 2, 3], 1)

    # Search with wrong ef
    with pytest.raises(ValueError, match="ef_search"):
        hnsw.search([1, 2, 3, 4], k=1, ef_search=0)
