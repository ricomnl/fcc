"""Assorted utility functions."""

from pathlib import Path


def write_contacts(atom_pair_list, filepath):
    """Write pairs of atoms to a file, one pair per line.

    atom_pair_list is an iterable of (Atom, Atom) objects.
    """

    if isinstance(filepath, str):
        filepath = Path(filepath)
    elif isinstance(filepath, Path):
        pass
    else:
        raise IOError("'filepath' must be of type str or pathlib.Path")

    # out_fn = f"{structure.filepath.stem}_contacts_{chain_i}_{chain_j}.txt"
    out_str = [f"{atom_i}\t{atom_j}" for atom_i, atom_j in atom_pair_list]
    with filepath.open("wt") as handle:
        handle.write("\n".join(out_str))
