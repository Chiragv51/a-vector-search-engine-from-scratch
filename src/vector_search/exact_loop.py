"""
Pure Python loop-based exact KNN search.
Acts as the immutable ground-truth baseline for all subsequent optimizations.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .metrics import squared_l2


@dataclass(frozen=True)
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray


class ExactLoopIndex:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._vectors: list[np.ndarray] = []
        self._ids: list[int] = []

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, vector, item_id: int) -> None:
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self.dimension,):
            raise ValueError(f"expected shape {(self.dimension,)}, got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("vectors must be finite")
        if item_id in self._ids:
            raise ValueError(f"duplicate item_id: {item_id}")
        self._vectors.append(v.copy())
        self._ids.append(item_id)

    def search(self, query, k: int) -> SearchResult:
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dimension,):
            raise ValueError(f"expected query shape {(self.dimension,)}, got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("query must be finite")
        if not 1 <= k <= len(self._vectors):
            raise ValueError(f"k must be between 1 and {len(self._vectors)}, got {k}")
        
        scored: list[tuple[float, int]] = []
        for item_id, vector in zip(self._ids, self._vectors):
            dist = squared_l2(q, vector)
            scored.append((dist, item_id))
        
        # Deterministic tie-breaking on (distance, item_id)
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        top = scored[:k]
        
        return SearchResult(
            ids=np.asarray([item_id for _, item_id in top], dtype=np.int64),
            distances=np.asarray([distance for distance, _ in top], dtype=np.float32),
        )
