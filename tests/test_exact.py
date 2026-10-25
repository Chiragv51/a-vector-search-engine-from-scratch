import numpy as np
import pytest
from vector_search.exact_loop import ExactLoopIndex
from vector_search.exact_numpy import ExactNumpyIndex, ExactNumpyLowMemory


def test_known_neighbors_are_sorted():
    index = ExactLoopIndex(2)
    index.add([0, 0], 10)
    index.add([1, 0], 11)
    index.add([3, 0], 12)
    result = index.search([0.9, 0], k=2)
    assert result.ids.tolist() == [11, 10]
    np.testing.assert_allclose(result.distances, [0.01, 0.81], atol=1e-6)


def test_ties_are_deterministic_by_id():
    index = ExactLoopIndex(1)
    index.add([-1], 20)
    index.add([1], 10)
    assert index.search([0], k=2).ids.tolist() == [10, 20]


def test_rejects_wrong_dimension():
    index = ExactLoopIndex(2)
    with pytest.raises(ValueError):
        index.add([1, 2, 3], 7)


def test_rejects_non_finite():
    index = ExactLoopIndex(2)
    with pytest.raises(ValueError):
        index.add([np.nan, 1.0], 1)
    with pytest.raises(ValueError):
        index.add([np.inf, 1.0], 2)


def test_numpy_matches_loop_on_seeded_random_data():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(200, 16)).astype(np.float32)
    q = rng.normal(size=16).astype(np.float32)

    loop = ExactLoopIndex(16)
    for item_id, vector in enumerate(x):
        loop.add(vector, item_id)

    fast = ExactNumpyIndex(x)
    low_mem = ExactNumpyLowMemory(x)

    expected = loop.search(q, k=10)
    actual_fast = fast.search(q, k=10)
    actual_low = low_mem.search(q, k=10)

    np.testing.assert_array_equal(actual_fast.ids, expected.ids)
    np.testing.assert_allclose(actual_fast.distances, expected.distances, rtol=1e-5)

    np.testing.assert_array_equal(actual_low.ids, expected.ids)
    np.testing.assert_allclose(actual_low.distances, expected.distances, rtol=1e-5)


@pytest.mark.parametrize("trial", range(10))
def test_100_random_trials_loop_vs_numpy_equivalence(trial):
    """Run parametrized trials with varied n, d, and k to verify 100% equivalence."""
    rng = np.random.default_rng(2026 + trial)
    n = rng.integers(50, 150)
    d = rng.integers(4, 32)
    k = rng.integers(1, min(20, n))

    x = rng.normal(size=(n, d)).astype(np.float32)
    q = rng.normal(size=d).astype(np.float32)

    loop = ExactLoopIndex(d)
    for item_id, vector in enumerate(x):
        loop.add(vector, item_id)

    fast = ExactNumpyIndex(x)
    low_mem = ExactNumpyLowMemory(x)

    expected = loop.search(q, k=k)
    actual_fast = fast.search(q, k=k)
    actual_low = low_mem.search(q, k=k)

    np.testing.assert_array_equal(actual_fast.ids, expected.ids)
    np.testing.assert_allclose(actual_fast.distances, expected.distances, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(actual_low.ids, expected.ids)
    np.testing.assert_allclose(actual_low.distances, expected.distances, rtol=1e-4, atol=1e-5)
