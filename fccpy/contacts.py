"""Code to calculate atomic contacts."""

from itertools import combinations
from pathlib import Path

from numba import jit
import numpy as np

from fccpy.structure import Atom, atom_to_csv
from fccpy.utils import as_file_handle


__all__ = ["get_intermolecular_contacts", "contacts_to_file", "contacts_from_file"]


# Functions to calculate contacts
@jit(
    "UniTuple(int64[:], 2)(f8[:, :], i8[:], i8[:], f8)",
    nopython=True,
    nogil=True,
    fastmath=True,
)
def get_pairwise_contacts(coordinates, idx_a, idx_b, dmax):
    """Identify pairs of atoms in `coordinates` are within `dmax` distance.

    Distances are calculated in Euclidean space (3D).

    Parameters
    ----------
    coordinates : numpy.ndarray
        2D array of shape (N, 3) with positions (`float`) for N atoms.
    idx_a, idx_b : numpy.ndarray
        1D array of `int` with indices corresponding to rows in `coordinates`.
        Contacts will be calculated only between coordinates of atoms in
        `idx_a` vs those in `idx_b`.
    dmax : float
        Maximum distance (in units of coordinates) to consider two atoms as a
        contact.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple with two arrays of `int` with the indexes of atoms in contact.
        Indexes correspond to the `coordinates` array.
    """

    # pre-allocate result array
    contacts_array = np.zeros((idx_a.shape[0], idx_a.shape[0]), dtype=np.bool_)

    # Slice coordinate arrays
    xyz_a = coordinates[idx_a]
    xyz_b = coordinates[idx_b]

    # run distance calculations
    d_max_sq = dmax * dmax
    for i, (xi, yi, zi) in enumerate(xyz_a):
        for j, (xj, yj, zj) in enumerate(xyz_b):
            dij_sq = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
            if dij_sq <= d_max_sq:
                contacts_array[i, j] = True

    # Convert indices to idx_* indices
    i_a, i_b = np.nonzero(contacts_array)
    return (idx_a[i_a], idx_b[i_b])


def get_intermolecular_contacts(structure, max_distance=5.0):
    """Return `Atom` pairs within `max_distance` in the input structure.

    Ignores contacts between `Atom` in the same chain.

    Parameters
    ----------
    structure : `fccpy.Structure`
        Object with 3D coordinates and atom information.
    max_distance : float
        Maximum distance to consider two atoms in contacts. Units must
        match those of the coordinates of the `structure` input.

    Yields
    ------
    tuple[`Atom`, `Atom`]
        Pairs of `Atom` instances in contact.
    """

    # Divide atoms in chains
    # Convert lists of atom idxs to numpy arrays
    atoms_per_chain = [
        np.array(v, dtype=np.int64) for v in structure.atoms_by_chain().values()
    ]

    # Calculate pairwise contacts between atoms of each pair of chains
    for idx_chain_i, idx_chain_j in combinations(atoms_per_chain, 2):
        pairs = get_pairwise_contacts(
            structure.xyz, idx_chain_i, idx_chain_j, max_distance
        )
        for i, j in zip(*pairs):
            yield structure.atom(i), structure.atom(j)


def contacts_to_file(atom_pairs, filepath):
    """Write pairs of atoms to file in CSV format.

    Parameters
    ----------
    atom_pairs : iterable[tuple[`Atom`, `Atom`]]
        List or iterator with pairs of atoms.
    filepath : Path
        Path to output file.
    """

    try:
        filepath = Path(filepath)
    except TypeError:
        raise IOError("'filepath' must be of type str or pathlib.Path")

    with as_file_handle(filepath, "wt") as handle:
        contents = (f"{atom_to_csv(a_j)},{atom_to_csv(a_j)}" for a_i, a_j in atom_pairs)
        handle.write("\n".join(contents))


def contacts_from_file(filepath):
    """Read pairs of atoms from a file in CSV format.

    Parameters
    ----------
    filepath : Path
        Path to output file.

    Returns
    -------
    iterable[tuple[`Atom`, `Atom`]]
    """

    with as_file_handle(filepath, "rt") as handle:
        for lineno, line in enumerate(handle):
            try:
                (
                    chain_a,
                    resid_a,
                    icode_a,
                    name_a,
                    chain_b,
                    resid_b,
                    icode_b,
                    name_b,
                ) = line.rstrip().split(",")
            except ValueError:
                msg = f"Error parsing contact file {filepath.name} at line {lineno}"
                raise ValueError(msg)

            yield Atom(chain_a, resid_a, icode_a, name_a), Atom(
                chain_b, resid_b, icode_b, name_b
            )
