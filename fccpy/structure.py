"""Simple coordinate parsing code into objects."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fccpy.exceptions import StructureParserException
from fccpy.utils import open_file


__all__ = ["parse_pdb"]


# Objects to store atomic information
@dataclass(frozen=True, order=True)
class Atom:
    chain: str
    resid: int
    icode: str
    name: str

    def __str__(self):
        return f"{self.chain:<2s} {self.resid:>5d} {self.icode:<1s} {self.name:<4s}"


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


# Utility functions
def is_hydrogen(atom_fullname):
    """Return True is the atom is a hydrogen."""
    # TODO: try regex
    name = atom_fullname.strip()
    if atom_fullname[0].isalpha() and not atom_fullname[2:].isdigit():
        putative_elem = name
    elif name[0].isdigit():  # e.g. 1HE2
        putative_elem = name[1]
    else:
        putative_elem = name[0]
    return putative_elem == "H"


def divide_by_chain(structure):
    """Return a dictionary of atom indices grouped by chain id."""
    atoms_per_chain = defaultdict(list)
    for atom_idx, atom in enumerate(structure):
        atoms_per_chain[atom.chain].append(atom_idx)
    return atoms_per_chain


# IO functions
def parse_pdb(filepath, hetatm=False):
    """Read a PDB file into a Structure dataclass.

    If hetatm is False (default), ignores HETATM records.

    Raises exception if the structure has multiple MODELs.
    """

    try:
        filepath = Path(filepath)
    except TypeError:
        raise IOError("'filepath' must be of type str or pathlib.Path")

    fp = filepath.resolve(strict=True)

    if hetatm:
        rectypes = ("ATOM  ", "HETATM")
    else:
        rectypes = ("ATOM  ",)

    atoms = []
    coordinates = []
    with open_file(fp) as handle:
        for ln, line in enumerate(handle, start=1):
            if line.startswith(rectypes):
                name = line[12:16]
                element = line[76:78].strip()
                if element == "H":
                    continue
                elif not element and is_hydrogen(name):
                    continue

                try:
                    chain_id = line[21]
                    resid_num = int(line[22:26])
                    icode = line[26]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    msg = f"Error parsing file '{filepath.name}' on line {ln}"
                    raise StructureParserException(msg) from None
                else:
                    atom = Atom(chain_id, resid_num, icode, name.strip())
                    atoms.append(atom)
                    coordinates.append((x, y, z))

            elif line.startswith(("MODEL ", "ENDMDL")):
                msg = "Multi-model structures are not supported."
                raise StructureParserException(msg)

    return Structure(filepath, atoms, np.asarray(coordinates, dtype=np.float64))
