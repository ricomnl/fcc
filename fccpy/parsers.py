"""IO functions: parsers and writers."""

from pathlib import Path

from Bio.PDB import PDBParser
import numpy as np

from fccpy.exceptions import StructureParserException
from fccpy.structure import Atom, Structure
from fccpy.utils import as_file_handle


__all__ = ["read_pdb"]


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


# IO functions
def read_pdb(filepath, hetatm=False):
    """Read a PDB file into a Structure dataclass.

    If hetatm is False (default), ignores HETATM records. Raises
    exception if the structure has multiple MODELs.

    Parameters
    ----------
    filepath : Path or str
        Path to input structure to parse.
    hetatm : bool, optional
        Flag to indicate if HETATM records should be read or not.
        Default is `False`.

    Returns
    -------
    `fccpy.Structure`

    Raises
    ------
    IOError
        If the input file could not be read or cannot be converted
        to a `Path`.
    StructureParserException
        If the PDB file could not be parsed for some reason.
    """

    _Atom = Atom  # bring to local scope as they are called freq.
    _is_hydrogen = is_hydrogen

    parser = PDBParser(PERMISSIVE=1)
    fp = filepath.resolve(strict=True)
    structure = parser.get_structure("structure", as_file_handle(fp, "rt"))
    
    if len([*structure.get_models()]) > 1:
        msg = "Multi-model structures are not supported."
        raise StructureParserException(msg)

    atoms = []
    xyz = []
    for atom_line in structure.get_atoms():
        name = atom_line.get_name()
        element = atom_line.element
        atom_idx = atom_line.get_serial_number()
        _, _, chain_id, (het, residue_id, icode), _ = atom_line.get_full_id()
        x, y, z = [*atom_line.get_coord()]

        if element == "H":
            continue
        elif not element and _is_hydrogen(name):
            continue

        if not hetatm and het.strip() != "":
            continue
        
        atom = _Atom(chain_id, atom_idx, residue_id, icode, name)
        atoms.append(atom)
        xyz.append((x, y, z))        

    return Structure(filepath, atoms, np.asarray(xyz, dtype=np.float64))
