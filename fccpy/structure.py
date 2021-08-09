"""Objects and utility functions to store/manipulate coordinate data."""

from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["filter_by_chain"]


# Objects to store atomic information
Atom = namedtuple("Atom", ["chain", "resid", "icode", "name"])


@dataclass(frozen=True)
class Structure:
    filepath: Path
    atoms: list[Atom]
    xyz: np.ndarray

    def __repr__(self):
        return f"{self.filepath.name} ({len(self.atoms)} atoms)"

    def __iter__(self):
        yield from self.atoms

    def atom(self, idx):
        """Return a specific Atom."""
        return self.atoms[idx]

    def atoms_by_chain(self):
        """Return a dictionary of atom indices grouped by chain id."""
        atoms_per_chain = defaultdict(list)
        for atom_idx, atom in enumerate(self.atoms):
            atoms_per_chain[atom.chain].append(atom_idx)
        return atoms_per_chain


# Utility functions
def atom_to_csv(self):
    return f"{self.chain},{self.resid},{self.icode},{self.name}"


# Filter functions
def filter_by_chain(structure, chain_ids):
    """Return a Structure containing only atoms belonging to specific chains.

    Parameters
    ----------
    structure : fccpy.Structure
        Structure object to filter.
    chain_ids : list[str]
        List of chain identifiers.

    Returns
    -------
    fccpy.Structure

    Raises
    ------
    ValueError
        If chain_ids is empty or if no atoms match the requested chains.
    """
    if not chain_ids:
        raise ValueError("Argument `chain_ids` must not be empty.")

    include_set = set(chain_ids)
    sele = ((i, a) for i, a in enumerate(structure) if a.chain in include_set)
    sele_idx, sele_atoms = zip(*sele)

    if not sele_idx:
        raise ValueError("Selection returned zero atoms.")

    return Structure(
        filepath=structure.filepath, atoms=sele_atoms, xyz=structure.xyz[sele_idx]
    )
