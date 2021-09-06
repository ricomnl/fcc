"""Tests for fccpy.similarity module."""

from pathlib import Path
import tempfile

import numpy as np

from fccpy.similarity import jaccard, overlap, fcc
from fccpy.similarity import _is_symmetric
from fccpy.similarity import _build_matrix_asym, _build_matrix_sym, build_matrix
from fccpy.similarity import read_matrix, write_matrix

# Shared variables
setA = {"a", "b", "c"}
setB = {"a", "b", "c", "d"}
setC = {"b", "c", "d", "e"}


# Test metrics
def test_jaccard():
    """Test jaccard similarity metric."""

    sAB = jaccard(setA, setB)
    sBA = jaccard(setB, setA)

    assert sAB == sBA == 0.75


def test_overlap():
    """Test overlap similarity metric."""

    sAB = overlap(setA, setB)
    sBA = overlap(setB, setA)

    assert sAB == sBA == 1.0


def test_fcc():
    """Test fcc similarity metric."""

    sAB = fcc(setA, setB)
    sBA = fcc(setB, setA)

    assert sAB != sBA
    assert sAB == 1.0
    assert sBA == 0.75


def test_is_symmetric():
    """Test _is_symmetric private function."""

    assert _is_symmetric(jaccard) is True
    assert _is_symmetric(overlap) is True
    assert _is_symmetric(fcc) is False


# Test matrix builders
def test_build_matrix_sym():
    """Test build_matrix_sym private function."""

    setlist = [setA, setB, setC]

    mtx = list(_build_matrix_sym(setlist))
    mtx_fcc = list(_build_matrix_sym(setlist, metric=fcc))

    assert len(mtx) == 3
    assert mtx == mtx_fcc  # test default option
    assert mtx == [(0, 1, 1.0, 1.0), (0, 2, 2 / 3, 2 / 3), (1, 2, 0.75, 0.75)]


def test_build_matrix_asym():
    """Test build_matrix_asym private function."""

    setlist = [setA, setB, setC]

    mtx = list(_build_matrix_asym(setlist))
    mtx_fcc = list(_build_matrix_asym(setlist, metric=fcc))

    assert len(mtx) == 3
    assert mtx == mtx_fcc  # test default option
    assert mtx == [(0, 1, 1.0, 0.75), (0, 2, 2 / 3, 0.5), (1, 2, 0.75, 0.75)]


def test_build_matrix():
    """Test build_matrix()."""

    def _compare_lists_elementwise(lA, lB):
        for eleA, eleB in zip(lA, lB):
            if isinstance(eleA, list) and isinstance(eleB, list):
                _compare_lists_elementwise(eleA, eleB)
            else:
                assert abs(eleA - eleB) < 0.01

    setlist = [setA, setB, setC]

    idxs, sims = build_matrix(setlist, metric=fcc)
    expected_idxs = [[0, 1], [0, 2], [1, 2]]
    expected_sims = [[1.0, 0.75], [0.66, 0.5], [0.75, 0.75]]
    assert len(sims) == 3
    assert idxs.tolist() == expected_idxs
    _compare_lists_elementwise(sims.tolist(), expected_sims)

    idxs, sims = build_matrix(setlist, metric=jaccard)
    expected_sims = [[0.75, 0.75], [0.4, 0.4], [0.6, 0.6]]
    assert len(sims) == 3
    assert idxs.tolist() == expected_idxs
    _compare_lists_elementwise(sims.tolist(), expected_sims)

    _, sims = build_matrix(setlist, metric=overlap)
    expected_sims = [[1.0, 1.0], [0.66, 0.66], [0.75, 0.75]]
    assert len(sims) == 3
    assert idxs.tolist() == expected_idxs
    _compare_lists_elementwise(sims.tolist(), expected_sims)


# IO
def test_read_write_matrix():
    """Test roundabout read/write matrix files."""

    setlist = [setA, setB, setC]
    idxs, sims = build_matrix(setlist)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir, "tmp.out")
        write_matrix(idxs, sims, fp)
        idxs_, sims_ = read_matrix(fp)

        assert np.allclose(idxs, idxs_)
        assert np.allclose(sims, sims_)
