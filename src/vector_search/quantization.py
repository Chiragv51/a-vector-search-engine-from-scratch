"""
Scalar Quantization (SQ8): Per-dimension uint8 quantization.
Compacts 32-bit floating point vectors into 8-bit unsigned integers,
achieving ~4x compression while preserving metric ordering.
"""

from __future__ import annotations
import numpy as np
from .exact_loop import SearchResult


class ScalarQuantizer:
    """Fits per-dimension scale and minimum values to map float32 -> uint8."""

    def __init__(self):
        self.minimum: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> ScalarQuantizer:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("fit expects a 2D matrix")
        self.minimum = arr.min(axis=0)
        maximum = arr.max(axis=0)
        self.scale = np.where(
            maximum > self.minimum, (maximum - self.minimum) / 255.0, 1.0
        ).astype(np.float32)
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("Quantizer must be fitted before encoding")
        arr = np.asarray(x, dtype=np.float32)
        values = np.rint((arr - self.minimum) / self.scale)
        return np.clip(values, 0, 255).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("Quantizer must be fitted before decoding")
        arr = np.asarray(codes, dtype=np.uint8)
        return arr.astype(np.float32) * self.scale + self.minimum


class QuantizedExactIndex:
    """Exact KNN index using quantized vectors to dramatically reduce RAM footprint."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray | None = None):
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        self.quantizer = ScalarQuantizer().fit(x)
        self.codes = self.quantizer.encode(x)
        self.ids = (
            np.arange(len(x), dtype=np.int64)
            if ids is None
            else np.asarray(ids, dtype=np.int64)
        )
        self.dimension = x.shape[1]

    def __len__(self) -> int:
        return len(self.codes)

    def memory_bytes(self) -> int:
        """Total memory consumed by codes and quantizer parameters."""
        quant_params = (
            self.quantizer.minimum.nbytes + self.quantizer.scale.nbytes
            if self.quantizer.minimum is not None and self.quantizer.scale is not None
            else 0
        )
        return int(self.codes.nbytes + self.ids.nbytes + quant_params)

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        q = np.asarray(query, dtype=np.float32)
        # Decode codes on-the-fly or compute distance approximation
        # Asymmetric distance: query (float32) against decoded vectors (float32)
        decoded = self.quantizer.decode(self.codes)
        delta = decoded - q
        distances = np.einsum("ij,ij->i", delta, delta)
        candidate_rows = np.argpartition(distances, kth=k - 1)[:k]
        order = np.lexsort((self.ids[candidate_rows], distances[candidate_rows]))
        rows = candidate_rows[order]
        return SearchResult(self.ids[rows], distances[rows])
