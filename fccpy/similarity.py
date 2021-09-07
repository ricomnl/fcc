"""Functions to compute similarity matrices."""

from itertools import combinations

import h5py
import numpy as np

from fccpy.exceptions import MatrixReadingException
from fccpy.utils import as_file_handle


# Similarity metrics
# Defined as a bundle after the last one.
def jaccard(set_A, set_B):  # fix latex formula
    r"""Calculate the Jaccard similarity between two sets.

    .. math:: D(A, B) = \frac{|{A \cap B}|}{|{A \cap B}|}

    Its value is always bounded between 0 (no elements in common)
    and 1 (all elements shared).
    """
    return len(set_A & set_B) / len(set_A | set_B)


def overlap(set_A, set_B):
    r"""Calculate the Overlap coefficient between two sets.

    .. math:: D(A, B) = \frac{|{A \cap B}|}{min(|{A}|, |{B}|}

    Similar to Jaccard. Its value is always bounded between 0
    (no elements in common) and 1 (if one of the sets if a subset
    of the other).
    """

    return len(set_A & set_B) / min(len(set_A), len(set_B))


def fcc(set_A, set_B):
    r"""Calculate the FCC similarity between two sets.

    .. math:: D(A, B) = \frac{|{A \cap B}|}{|{A}|}

    This is potentially an asymmetric similarity metric, since the
    lengths of A and B aren't necessarily equal. Its value is always
    bounded between 0 (no elements in common) and 1 (all elements shared).
    """

    return len(set_A & set_B) / len(set_A)


# Matrix builders (wrappers for the functions above)
def build_matrix(setlist, metric=fcc, precision=np.half):
    """Build a similarity matrix from sets of hashed contacts.

    The output is a pair of arrays representing the elements of
    the similarity matrix that are non-zero. The first array is
    the coordinates of the elements (i, j), and the second array
    contains the similarities s(i,j) and s(j,i), which might not
    be the same depending on the metric used.

    The rationale for not storing the similarity matrix in this
    format is two-fold: 1) memory-efficiency, since we only store
    elements that contain any information, and 2) performance, since
    for clustering pairs of elements with similarity = 0 are irrelevant.

    Parameters
    ----------
    setlist : [[int]]
        a list of sets of integers (hashes), each representing a unique
        contact between two entities (atoms, residues, chains, etc).

    metric : callable
        a function that takes two sets and produces a tuple of similarity
        values (integers) with the similarities of AB and BA. This module
        implements `fcc`, `jaccard` and `overlap`.

    precision : np.dtype
        precision in which to store the similarities values in the resulting
        matrix. Default is `np.half`.

    Returns
    -------
    idxs : np.ndarray
        array of indices to the original setlist, for each pair of elements
        for which similarities were computed.
    sims : np.ndarray
        array of similarities between pairs of elements i and i of shape (n, 2),
        where n is the number of pairwise combinations of all N elements of the
        original setlist where s(i, j) or s(j, i) is non-zero.
    """

    if _is_symmetric(metric):
        _builder = _build_matrix_sym
    else:
        _builder = _build_matrix_asym

    idxs_, sims_ = [], []
    for (i, j, s_ij, s_ji) in _builder(setlist, metric=metric):
        # Ignore zeros to save space
        if s_ij or s_ji:
            idxs_.append((i, j))
            sims_.append((s_ij, s_ji))

    idxs_ = np.array(idxs_, dtype=np.int32)
    sims_ = np.array(sims_, dtype=precision)

    return idxs_, sims_


def _is_symmetric(f):
    """Private function to help users not shoot their feet.

    Calculates f(AB) and f(BA) and returns True if they are the
    same (f() is symmetric), False otherwise (f() is asymmetric).
    """

    A = {1, 2, 3, 4, 5}
    B = {1, 2, 3}
    return f(A, B) == f(B, A)


def _build_matrix_asym(setlist, metric=fcc):
    """Build a symmetric similarity matrix from sets of hashed contacts.

    Parameters and return are the same as the original `build_matrix`
    function, which you should use since the performance hit of the
    wrapper is minimal.
    """

    metric = metric
    all_pairs = combinations(enumerate(setlist), 2)
    for (i, set_i), (j, set_j) in all_pairs:
        s_ij = metric(set_i, set_j)
        s_ji = metric(set_j, set_i)
        yield i, j, s_ij, s_ji


def _build_matrix_sym(setlist, metric=overlap):
    """Build a symmetric similarity matrix from sets of hashed contacts.

    Parameters and return are the same as the original `build_matrix`
    function, which you should use since the performance hit of the
    wrapper is minimal. This variant exists for performance, since we
    don't need to calculate f(AB) and f(BA) when f() is symmetric.

    WARNING: This function does not check if `metric` is in fact symmetric.
    That is up to you to know... Garbage in, garbage out.
    """

    metric = metric
    all_pairs = combinations(enumerate(setlist), 2)
    for (i, set_i), (j, set_j) in all_pairs:
        s_ij = metric(set_i, set_j)
        yield i, j, s_ij, s_ij


# IO
def read_matrix_csv(filepath, precision=np.half):
    """Read a CSV-formatted similarity matrix from disk.

    Matrix must have the form:
    i, j, s_ij, s_ji
        where s_ij, s_ji are the similarity values between two
        structures rank i and j.

    Parameters
    ----------
    filepath : pathlib.Path or file-like object
        Path to, or an open file handle to a matrix file.

    precision : np.dtype
        precision in which to store the similarities values in the resulting
        matrix. Default is `np.half`.

    Returns
    -------
    idxs : np.ndarray
        array of indices to the original setlist, for each pair of elements
        for which similarities were computed.
    sims : np.ndarray
        array of similarities between pairs of elements i and i of shape (n, 2),
        where n is the number of pairwise combinations of all N elements of the
        original setlist where s(i, j) or s(j, i) is non-zero.
    """

    idxs, sims = [], []
    with as_file_handle(filepath, "rt") as handle:
        for ln, line in enumerate(handle, start=1):
            try:
                i, j, s_ij, s_ji = line.strip().split(",")
                i, j = int(i), int(j)
                s_ij, s_ji = float(s_ij), float(s_ji)
            except ValueError as err:
                msg = f"Error parsing line {ln} from {filepath.name}: {err}"
                raise MatrixReadingException(msg) from None
            else:
                idxs.append((i, j))
                sims.append((s_ij, s_ji))

    idxs = np.array(idxs, dtype=np.int32)
    sims = np.array(sims, dtype=precision)
    return idxs, sims


def read_matrix(filepath, precision=np.half):
    """Read a similarity matrix from an HDF file on disk.

    Will read from a dataset "mtx" with shape (n, 2), where n is
    the number of elements in the matrix. Values of the dataset
    are similarities between structures i and j: s_ij, s_ji.

    Parameters
    ----------
    filepath : pathlib.Path or file-like object
        Path to, or an open file handle to a matrix file.

    precision : np.dtype
        precision in which to store the similarities values in the resulting
        matrix. Default is `np.half`.

    Returns
    -------
    idxs : np.ndarray
        array of indices to the original setlist, for each pair of elements
        for which similarities were computed.
    sims : np.ndarray
        array of similarities between pairs of elements i and i of shape (n, 2),
        where n is the number of pairwise combinations of all N elements of the
        original setlist where s(i, j) or s(j, i) is non-zero.
    """

    with as_file_handle(filepath, "rb") as handle:
        with h5py.File(handle, "r") as h5f:
            idxs = np.array(h5f["idxs"][:, :], dtype=np.int32)
            sims = np.array(h5f["sims"][:, :], dtype=precision)

    assert (
        idxs.shape[1] == 2
    ), f"Indexes array must be of shape (n, 2) but is {idxs.shape}."

    assert (
        sims.shape[1] == 2
    ), f"Similarities array must be of shape (n, 2) but is {sims.shape}."

    assert idxs.size == sims.size, (
        "Indexes and similarities arrays must be of same length: "
        f"{idxs.size} != {sims.size}"
    )

    return idxs, sims


def write_matrix_csv(idxs, sims, filepath, precision=3):
    """Write a similarity matrix to disk in CSV format.

    This is intended only for debugging purposes, since reading and writing
    as CSV is 1) much slower and 2) takes a lot more space.

    Matrix will have the form:
    i, j, s_ij, s_ji
        where s_ij, s_ji are the similarity values between sets i and j.

    Parameters
    ----------
    idxs : np.ndarray
        array of indices to the original setlist, for each pair of elements
        for which similarities were computed.
    sims : np.ndarray
        array of similarities between pairs of elements i and i of shape (n, 2),
        where n is the number of pairwise combinations of all N elements of the
        original setlist where s(i, j) or s(j, i) is non-zero.
    filepath : pathlib.Path
        Path to output file.
    precision : float
        number of decimal places to use for writing similarity values. Default
        is 3.
    """

    with as_file_handle(filepath, "wt") as handle:
        idxs, sims = idxs.tolist(), sims.tolist()
        for (i, j), (s_ij, s_ji) in zip(idxs, sims):
            print(f"{i},{j},{s_ij:.{precision}},{s_ji:.{precision}}", file=handle)


def write_matrix(idxs, sims, filepath):
    """Write a similarity matrix to disk in HDF5 format.

    HDF5 file will have two datasets:
        - "idxs", to store the idxs array containing indices i and j.
        - "sims", to store the sims array containing the values of
            similarities s(i,j) and s(j,i) for each pair of elements
            i and j.

    Parameters
    ----------
    idxs : np.ndarray
        array of indices to the original setlist, for each pair of elements
        for which similarities were computed.
    sims : np.ndarray
        array of similarities between pairs of elements i and i of shape (n, 2),
        where n is the number of pairwise combinations of all N elements of the
        original setlist where s(i, j) or s(j, i) is non-zero.
    filepath : pathlib.Path
        Path to output file.
    """

    with as_file_handle(filepath, "wb") as handle:
        with h5py.File(handle, "w") as h5f:
            h5f.create_dataset("idxs", data=idxs, compression="gzip")
            h5f.create_dataset("sims", data=sims, compression="gzip")
