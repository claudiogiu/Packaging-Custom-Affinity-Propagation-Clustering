"""
Testing for Affinity Propagation Algorithm.

"""

import numpy as np

import pytest

from affinityprop import AffinityPropagation


def test_output_shapes():
    # Check that labels and cluster centers have correct shape
    S = -np.abs(np.random.randn(6, 6))
    np.fill_diagonal(S, -0.5)
    model = AffinityPropagation(preference=-1.0)
    model.fit(S)

    assert model.labels_.shape == (6,)
    assert isinstance(model.cluster_centers_indices_, np.ndarray)


def test_creates_at_least_one_cluster():
    # Check that at least one cluster is detected
    S = -np.abs(np.random.rand(4, 4))
    np.fill_diagonal(S, -2.0)
    model = AffinityPropagation(preference=-3.0)
    model.fit(S)

    assert len(np.unique(model.labels_)) >= 1


def test_invalid_input_shape_raises():
    # Check that non-square input raises a ValueError
    S = np.random.randn(5, 3)
    model = AffinityPropagation()
    with pytest.raises(ValueError):
        model.fit(S)


def test_two_point_clustering():
    # Check that clustering works on a 2x2 similarity matrix
    S = np.array([[0.0, -1.0], [-1.0, 0.0]])
    model = AffinityPropagation(preference=-2.0)
    model.fit(S)

    assert model.labels_.shape == (2,)


def test_convergence_stability():
    # Check that the model converges and returns labels
    S = -np.abs(np.random.rand(8, 8))
    np.fill_diagonal(S, -1.0)
    model = AffinityPropagation(convergence_iter=2, max_iter=50)
    model.fit(S)

    assert model.labels_ is not None


def test_every_cluster_has_points():
    # Check that no cluster is left without assigned points
    S = -np.abs(np.random.rand(10, 10))
    np.fill_diagonal(S, -2.0)
    model = AffinityPropagation(preference=-3.0)
    model.fit(S)

    counts = np.bincount(model.labels_)
    assert np.all(counts > 0)