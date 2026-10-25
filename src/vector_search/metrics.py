"""
Distance and similarity metrics for vector search.
"""

from __future__ import annotations
import numpy as np


def squared_l2(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Calculate squared Euclidean (L2) distance between two 1D vectors.

    Using squared L2 distance avoids expensive monotonic square root calculations,
    producing an identical ranking order.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"expected equal 1D shapes, got {a.shape} and {b.shape}")
    delta = a - b
    return float(delta @ delta)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize each row of a 2D matrix to unit L2 norm."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cosine similarity is undefined for zero vectors")
    return x / norms


def cosine_similarity(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"expected equal 1D shapes, got {a.shape} and {b.shape}")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine similarity is undefined for zero vectors")
    return float((a @ b) / (norm_a * norm_b))
