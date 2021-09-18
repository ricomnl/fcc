"""Code to calculate atomic contacts."""

from collections import Counter
from itertools import combinations
from operator import attrgetter
from pathlib import Path

from numba import jit
import numpy as np

from fccpy.structure import Atom, atom_to_csv
from fccpy.utils import as_file_handle


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
        np.array(v, dtype=np.int64) for v in structure.atoms_by_chain.values()
    ]

    # Calculate pairwise contacts between atoms of each pair of chains
    for idx_chain_i, idx_chain_j in combinations(atoms_per_chain, 2):
        pairs = get_pairwise_contacts(
            structure.xyz, idx_chain_i, idx_chain_j, max_distance
        )
        for i, j in zip(*pairs):
            yield structure.atoms[i], structure.atoms[j]


# Selectors
BY_CHAIN = attrgetter("chain")
BY_RESIDUE = attrgetter("chain", "resid", "icode")
BY_RESIDUE_NOCHAIN = attrgetter("resid", "icode")
BY_ATOM = attrgetter("chain", "resid", "icode", "name")


# Hash functions
def hash_contact(atom_pair, selector1=None, selector2=None):
    """Return a unique hash for a pair of atoms.

    Parameters
    ----------
    atom_pair : tuple[`Atom`, `Atom`]
        A pair of atom objects.
    selector1, selector2 : callable, optional
        a callable that takes an `Atom` as input and returns
        some of its attributes. If a selectors is None, all
        attributes of the Atom will be considered.

    Returns
    -------
    a unique hash string to the input atom pair.

    Examples
    --------
    >>> from fccpy.contacts import BY_CHAIN
    >>> a1 = Atom("A", 1, "", "N")
    >>> a2 = Atom("B", 2, "", "N")
    >>> hash_contact(a1, a2)
    >>> hash_contact(a1, a2, selector1=BY_CHAIN, selector2=BY_CHAIN)
    """

    selector1 = selector1 if selector1 is not None else tuple
    selector2 = selector2 if selector2 is not None else tuple

    atom1, atom2 = atom_pair

    return str(hash((selector1(atom1), selector2(atom2))))


def hash_many(list_of_pairs, unique=True, selector1=None, selector2=None):
    """Run hash function on a list of contacts.

    Calls `hash_contact` on each item in the input list.

    Parameters
    ----------
    list_of_pairs : list[tuple[`Atom`, `Atom`]]
        a list of Atom pairs
    unique : bool, optional
        if `unique` is True (default), each unique pair
        in a list will be added once to the output set.
        If `unique` is False, copies of the same contact
        will be added. This is useful when the number of
        occurences of a pair, besides their identity, is
        necessary for further calculations.
    selector1, selector2 : callable, optional
        a callable that takes an `Atom` as input and returns
        some of its attributes. If a selector is `None`, will
        hash based on all attributes of the `Atom`.

    Returns
    -------
    a list of frozensets with hashes for each list of pairs.
    """

    hashes = Counter(
        hash_contact(pair, selector1=selector1, selector2=selector2)
        for pair in list_of_pairs
    )
    if unique:
        return frozenset(hashes)

    # hashA_0, hashA_1, hashB_0, ...
    return frozenset(f"{hash_}_{i}" for hash_, n in hashes.items() for i in range(n))


# IO
def read_contacts(filepath):
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

            atomA = Atom(chain_a, int(resid_a), icode_a, name_a)
            atomB = Atom(chain_b, int(resid_b), icode_b, name_b)
            yield atomA, atomB


def write_contacts(atom_pairs, filepath):
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
        contents = (f"{atom_to_csv(a_i)},{atom_to_csv(a_j)}" for a_i, a_j in atom_pairs)
        handle.write("\n".join(contents))
