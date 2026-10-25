import numpy as np
import pytest
from vector_search.quantization import ScalarQuantizer, QuantizedExactIndex
from vector_search.exact_numpy import ExactNumpyIndex
from vector_search.datasets import synthetic_dataset, recall_at_k


def test_scalar_quantizer_reconstruction_error():
    rng = np.random.default_rng(2026)
    x = rng.uniform(low=-10.0, high=10.0, size=(100, 16)).astype(np.float32)

    sq = ScalarQuantizer().fit(x)
    codes = sq.encode(x)
    assert codes.dtype == np.uint8
    assert codes.shape == (100, 16)

    decoded = sq.decode(codes)
    # The maximum quantization error per coordinate should be at most scale / 2
    max_err = np.max(np.abs(x - decoded))
    max_scale = np.max(sq.scale)
    assert max_err <= (max_scale / 2.0 + 1e-4)


def test_quantized_index_preserves_high_recall():
    n, d, query_count = 1000, 32, 50
    vectors, queries = synthetic_dataset(n, d, query_count, seed=2026)

    # Float32 exact truth
    exact_idx = ExactNumpyIndex(vectors)
    truth = np.stack([exact_idx.search(q, 10).ids for q in queries])

    # Quantized search
    quant_idx = QuantizedExactIndex(vectors)
    approx = np.stack([quant_idx.search(q, 10).ids for q in queries])

    recall = recall_at_k(truth, approx)
    # SQ8 typically preserves >98% recall on normal distributions
    assert recall >= 0.95, f"Expected high recall with SQ8, got {recall}"

    # Verify memory compression
    float_bytes = vectors.nbytes
    quant_bytes = quant_idx.memory_bytes()
    assert quant_bytes < float_bytes * 0.4  # Well over 2.5x to 3.5x compression including IDs
