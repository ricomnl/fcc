"""Objects and utility functions to store/manipulate coordinate data."""

from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Objects to store atomic information
Atom = namedtuple("Atom", ["chain", "resid", "icode", "name"])


@dataclass(frozen=True)
class Structure:
    """Dataclass to store atomic structure data."""

    filepath: Path
    atoms: list[Atom]
    xyz: np.ndarray

    def __repr__(self):
        """Return nice representation of structure for print()."""
        return f"{self.filepath.name} ({len(self.atoms)} atoms)"

    def __iter__(self):
        """Iterate on atoms of the structure."""
        yield from self.atoms

    @property
    def atoms_by_chain(self):
        """Return a dictionary of atom indices grouped by chain id."""
        atoms_per_chain = defaultdict(list)
        for atom_idx, atom in enumerate(self.atoms):
            atoms_per_chain[atom.chain].append(atom_idx)
        return atoms_per_chain


# Utility functions
def atom_to_csv(self):
    """Return Atom as a comma-separated string of its properties."""
    return f"{self.chain},{self.resid},{self.icode},{self.name}"
