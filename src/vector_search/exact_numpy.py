"""
Vectorized exact KNN search using NumPy broadcasting, einsum, and argpartition.
Also includes a low-memory variant using norm decomposition.
"""

from __future__ import annotations
import numpy as np
from .exact_loop import SearchResult


class ExactNumpyIndex:
    """Exact KNN index using vectorized NumPy operations."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray | None = None):
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("vectors must be a non-empty 2D matrix")
        if not np.all(np.isfinite(x)):
            raise ValueError("vectors must be finite")
        self.vectors = np.ascontiguousarray(x)
        self.ids = (
            np.arange(len(x), dtype=np.int64)
            if ids is None
            else np.asarray(ids, dtype=np.int64)
        )
        if self.ids.shape != (len(x),) or len(np.unique(self.ids)) != len(x):
            raise ValueError("ids must be unique and match vector count")

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def dimension(self) -> int:
        return self.vectors.shape[1]

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.vectors.shape[1],) or not np.all(np.isfinite(q)):
            raise ValueError("invalid query: shape mismatch or non-finite values")
        if not 1 <= k <= len(self.vectors):
            raise ValueError(f"invalid k: must be between 1 and {len(self.vectors)}")

        # Broadcast subtraction: (n, d) - (d,) -> (n, d)
        delta = self.vectors - q
        distances = np.einsum("ij,ij->i", delta, delta)

        # argpartition finds top-k in O(N) without full O(N log N) sort
        candidate_rows = np.argpartition(distances, kth=k - 1)[:k]

        # Fully sort only the selected k with ID as deterministic tie-breaker
        order = np.lexsort((self.ids[candidate_rows], distances[candidate_rows]))
        rows = candidate_rows[order]

        return SearchResult(self.ids[rows], distances[rows])


class ExactNumpyLowMemory(ExactNumpyIndex):
    """Low-memory exact KNN index.

    Avoids creating the giant (N, D) temporary subtraction matrix by expanding:
    ||x - q||^2 = ||x||^2 + ||q||^2 - 2(x . q)
    This saves gigabytes of memory on large vector databases.
    """

    def __init__(self, vectors: np.ndarray, ids: np.ndarray | None = None):
        super().__init__(vectors, ids)
        # Precompute ||x||^2 once at index creation
        self.row_norms = np.einsum("ij,ij->i", self.vectors, self.vectors)

    def all_distances(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32)
        q_norm = float(q @ q)
        # distances = ||x||^2 + ||q||^2 - 2 * (X @ q)
        distances = self.row_norms + q_norm - 2.0 * (self.vectors @ q)
        # Cancel potential small negative numerical roundoff
        return np.maximum(distances, 0.0)

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.vectors.shape[1],) or not np.all(np.isfinite(q)):
            raise ValueError("invalid query: shape mismatch or non-finite values")
        if not 1 <= k <= len(self.vectors):
            raise ValueError(f"invalid k: must be between 1 and {len(self.vectors)}")

        distances = self.all_distances(q)
        candidate_rows = np.argpartition(distances, kth=k - 1)[:k]
        order = np.lexsort((self.ids[candidate_rows], distances[candidate_rows]))
        rows = candidate_rows[order]

        return SearchResult(self.ids[rows], distances[rows])
