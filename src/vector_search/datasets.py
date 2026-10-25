"""
Dataset utilities and evaluation metrics for vector search.
"""

from __future__ import annotations
import numpy as np


def synthetic_dataset(
    n: int, d: int, query_count: int, seed: int = 2026
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible synthetic dataset and query vectors sampled from N(0, 1)."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    queries = rng.normal(size=(query_count, d)).astype(np.float32)
    return vectors, queries


def recall_at_k(exact_ids: np.ndarray, approx_ids: np.ndarray) -> float:
    """Calculate recall@k: the fraction of true nearest neighbors found by approximate search.

    Expects two 2D matrices of shape (query_count, k).
    """
    exact = np.asarray(exact_ids)
    approx = np.asarray(approx_ids)
    if exact.shape != approx.shape or exact.ndim != 2:
        raise ValueError(
            f"expected two query_count-by-k matrices of identical shape, got {exact.shape} and {approx.shape}"
        )
    k = exact.shape[1]
    if k == 0:
        return 0.0

    per_query = [
        len(set(e).intersection(set(a))) / k for e, a in zip(exact, approx)
    ]
    return float(np.mean(per_query))
