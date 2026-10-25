"""
Vector Search Engine from Scratch
=================================
A measurement-first, reproducible implementation of exact KNN,
vectorized NumPy KNN, educational HNSW, metadata filtering,
and scalar quantization with FAISS comparison.
"""

from .metrics import squared_l2, normalize_rows, cosine_similarity

__all__ = [
    "squared_l2",
    "normalize_rows",
    "cosine_similarity",
]
