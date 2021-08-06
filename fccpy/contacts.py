"""Functions to calculate contacts."""

from itertools import combinations

from numba import jit
import numpy as np

from fccpy.structure import divide_by_chain


__all__ = ["get_intermolecular_contacts"]


# Functions
@jit(
    "UniTuple(int64[:], 2)(f8[:, :], i8[:], i8[:], f8)",
    nopython=True,
    nogil=True,
    fastmath=True,
)
def get_pairwise_contacts(coordinates_array, idx_a, idx_b, max_dist):
    """Return indices of array element pairs within max_dist."""

    # pre-allocate result array
    contacts_array = np.zeros((idx_a.shape[0], idx_a.shape[0]), dtype=np.bool_)

    # Slice coordinate arrays
    xyz_a = coordinates_array[idx_a]
    xyz_b = coordinates_array[idx_b]

    # run distance calculations
    d_sq = max_dist * max_dist
    for i, (xi, yi, zi) in enumerate(xyz_a):
        for j, (xj, yj, zj) in enumerate(xyz_b):
            dij = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
            if dij <= d_sq:
                contacts_array[i, j] = True

    # Convert indices to idx_* indices
    i_a, i_b = np.nonzero(contacts_array)
    return (idx_a[i_a], idx_b[i_b])


def get_intermolecular_contacts(structure, max_dist=5.0):
    """Return a list of all atom-atom pairs within the max distance.

    Ignores contacts between atoms belonging to the same chain.
    """

    # Divide atoms in chains
    # Convert lists of ints to numpy arrays
    atoms_per_chain = {
        k: np.array(v, dtype=np.int64) for k, v in divide_by_chain(structure).items()
    }

    # Calculate pairwise contacts between atoms of each pair of chains
    for chain_i, chain_j in combinations(atoms_per_chain, 2):
        xyz_idx_i = atoms_per_chain[chain_i]
        xyz_idx_j = atoms_per_chain[chain_j]

        pairs = get_pairwise_contacts(structure.xyz, xyz_idx_i, xyz_idx_j, max_dist)

        for i, j in zip(*pairs):
            yield structure.atom(i), structure.atom(j)


# from pathlib import Path
# def write_contacts(atom_pair_list, filepath):
#     """Write pairs of atoms to a file, one pair per line.

#     atom_pair_list is an iterable of (Atom, Atom) objects.
#     """

#     if isinstance(filepath, str):
#         filepath = Path(filepath)
#     elif isinstance(filepath, Path):
#         pass
#     else:
#         raise IOError("'filepath' must be of type str or pathlib.Path")

#     # out_fn = f"{structure.filepath.stem}_contacts_{chain_i}_{chain_j}.txt"
#     out_str = [f"{atom_i}\t{atom_j}" for atom_i, atom_j in atom_pair_list]
#     with filepath.open("wt") as handle:
#         handle.write("\n".join(out_str))
