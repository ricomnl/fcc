"""Tests for fccpy.clustering module."""

import numpy as np

from fccpy.clustering import dbscan


def test_dbscan():
    """Test DBSCAN implementation."""

    # 5 elements: 2 clusters of 2 elements, 1 noise.
    idxs = [[0, 1], [0, 2], [0, 3], [0, 4], [2, 3], [3, 4]]
    sims = [(i, i) for i in [0.9, 0.2, 0.2, 0.2, 0.9, 0.2]]

    idxs = np.array(idxs)
    sims = np.array(sims)
    labels = dbscan(idxs, sims, eps=0.8, minsize=2)

    assert len(labels) == 4
    assert 4 not in labels
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
