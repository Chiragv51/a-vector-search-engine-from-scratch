"""
Integration tests for the Vector Search Engine.
Tests full end-to-end workflows connecting metrics, exact search,
HNSW indexing, metadata filtering, scalar quantization, and CLI demo execution.
"""

from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest

from vector_search.metrics import squared_l2, normalize_rows, cosine_similarity
from vector_search.exact_loop import ExactLoopIndex
from vector_search.exact_numpy import ExactNumpyIndex, ExactNumpyLowMemory
from vector_search.hnsw import HNSWIndex
from vector_search.filtering import (
    exact_filtered_search,
    post_filter_search,
    overfetch_filter_search,
    filter_aware_hnsw_search,
)
from vector_search.quantization import ScalarQuantizer, QuantizedExactIndex
from vector_search.datasets import synthetic_dataset, recall_at_k
from demo.cli import run_demo


class TestEndToEndPipeline:
    """Tests the full search lifecycle from raw vectors to retrieval."""

    def test_complete_vector_search_lifecycle(self):
        # 1. Generate data
        n, d, query_count = 300, 16, 20
        vectors, queries = synthetic_dataset(n=n, d=d, query_count=query_count, seed=123)

        # 2. Build exact ground truth indices
        loop_idx = ExactLoopIndex(d)
        for i, v in enumerate(vectors):
            loop_idx.add(v, i)

        numpy_idx = ExactNumpyIndex(vectors)
        lowmem_idx = ExactNumpyLowMemory(vectors)

        # 3. Build HNSW index
        hnsw_idx = HNSWIndex(dimension=d, M=8, ef_construction=64, seed=123)
        for i, v in enumerate(vectors):
            hnsw_idx.add(v, i)

        # 4. Build Quantized index
        quant_idx = QuantizedExactIndex(vectors)

        # 5. Execute queries and assert consistency
        for q in queries:
            k = 5
            res_loop = loop_idx.search(q, k=k)
            res_numpy = numpy_idx.search(q, k=k)
            res_lowmem = lowmem_idx.search(q, k=k)
            res_hnsw = hnsw_idx.search(q, k=k, ef_search=40)
            res_quant = quant_idx.search(q, k=k)

            # Strict equivalence between exact engines
            np.testing.assert_array_equal(res_loop.ids, res_numpy.ids)
            np.testing.assert_allclose(res_loop.distances, res_numpy.distances, rtol=1e-5)
            np.testing.assert_array_equal(res_numpy.ids, res_lowmem.ids)
            np.testing.assert_allclose(res_numpy.distances, res_lowmem.distances, rtol=1e-5)

            # Approximate and quantized results must return valid IDs and non-negative distances
            assert len(res_hnsw.ids) == k
            assert len(res_quant.ids) == k
            assert np.all(res_hnsw.distances >= 0)
            assert np.all(res_quant.distances >= 0)

            # HNSW and Quantized results should have significant overlap with ground truth
            hnsw_overlap = len(set(res_numpy.ids).intersection(set(res_hnsw.ids)))
            assert hnsw_overlap >= 3, f"Expected high recall, got overlap {hnsw_overlap}/5"

            quant_overlap = len(set(res_numpy.ids).intersection(set(res_quant.ids)))
            assert quant_overlap >= 4, f"Expected high recall from SQ8, got overlap {quant_overlap}/5"

    def test_cosine_similarity_normalized_search(self):
        """Verify that ranking by dot product on normalized vectors matches cosine similarity."""
        n, d = 200, 16
        vectors, queries = synthetic_dataset(n=n, d=d, query_count=10, seed=456)

        # Normalize both database vectors and queries
        norm_vectors = normalize_rows(vectors)
        norm_queries = normalize_rows(queries)

        exact_idx = ExactNumpyIndex(norm_vectors)

        for q_raw, q_norm in zip(queries, norm_queries):
            res = exact_idx.search(q_norm, k=5)
            # Verify that squared L2 on unit vectors correlates inversely with cosine similarity
            # ||a - b||^2 = 2 - 2*(a . b)
            for item_id, l2_dist in zip(res.ids, res.distances):
                cos_sim = cosine_similarity(q_norm, norm_vectors[item_id])
                expected_l2 = 2.0 - 2.0 * cos_sim
                np.testing.assert_allclose(l2_dist, expected_l2, atol=1e-5)

    def test_non_contiguous_array_slices(self):
        """Ensure indexes handle non-contiguous input arrays (e.g. strided slices)."""
        rng = np.random.default_rng(789)
        # Sliced matrix is non-contiguous in memory
        raw_matrix = rng.normal(size=(50, 32)).astype(np.float32)
        non_contiguous_vectors = raw_matrix[:, ::2]  # (50, 16) with stride 2
        assert not non_contiguous_vectors.flags.c_contiguous

        q = rng.normal(size=32).astype(np.float32)[::2]
        assert not q.flags.c_contiguous

        idx = ExactNumpyIndex(non_contiguous_vectors)
        res = idx.search(q, k=5)
        assert len(res.ids) == 5

        hnsw = HNSWIndex(dimension=16, M=4, ef_construction=16)
        for i, vec in enumerate(non_contiguous_vectors):
            hnsw.add(vec, i)
        res_hnsw = hnsw.search(q, k=5, ef_search=10)
        assert len(res_hnsw.ids) == 5

    def test_ground_truth_serialization_and_recall_calculation(self):
        """Test persisting ground truth to .npz and loading it back for recall calculation."""
        vectors, queries = synthetic_dataset(n=100, d=8, query_count=15, seed=999)
        exact_idx = ExactNumpyIndex(vectors)
        truth = np.stack([exact_idx.search(q, 10).ids for q in queries])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_truth.npz"
            np.savez_compressed(file_path, queries=queries, top10=truth, seed=999)

            with np.load(file_path) as loaded:
                loaded_top10 = np.array(loaded["top10"])
                loaded_queries = np.array(loaded["queries"])

            np.testing.assert_array_equal(loaded_top10, truth)
            np.testing.assert_array_equal(loaded_queries, queries)

            # Evaluating exact on itself should give recall = 1.0
            rec = recall_at_k(truth, loaded_top10)
            assert rec == 1.0

    def test_cli_demo_execution_no_error(self):
        """Verify the interactive CLI demo runs to completion without raising exceptions."""
        try:
            run_demo(n=100, d=16, k=3)
        except Exception as e:
            pytest.fail(f"demo.cli.run_demo crashed with exception: {e}")
