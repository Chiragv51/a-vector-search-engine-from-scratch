import numpy as np
import pytest
from vector_search.metrics import squared_l2, normalize_rows, cosine_similarity


def test_squared_l2_known_answer():
    assert squared_l2(np.array([1, 2]), np.array([4, 6])) == 25.0
    assert squared_l2([1, 2], [4, 6]) == 25.0


def test_squared_l2_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        squared_l2(np.array([1, 2]), np.array([1, 2, 3]))


def test_squared_l2_rejects_multidimensional():
    with pytest.raises(ValueError):
        squared_l2(np.array([[1, 2]]), np.array([[1, 2]]))


def test_normalized_rows_have_unit_norm():
    x = normalize_rows(np.array([[3, 4], [5, 12]], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-6)


def test_normalized_rows_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        normalize_rows(np.array([[0, 0], [1, 2]], dtype=np.float32))


def test_cosine_similarity_orthogonal_and_parallel():
    # Parallel
    np.testing.assert_allclose(cosine_similarity([1, 0], [5, 0]), 1.0, atol=1e-6)
    # Opposite
    np.testing.assert_allclose(cosine_similarity([1, 0], [-2, 0]), -1.0, atol=1e-6)
    # Orthogonal
    np.testing.assert_allclose(cosine_similarity([1, 0], [0, 1]), 0.0, atol=1e-6)
