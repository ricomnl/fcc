"""IO functions: parsers and writers."""

from pathlib import Path

import numpy as np

from fccpy.exceptions import StructureParserException
from fccpy.structure import Atom, Structure
from fccpy.utils import open_file


__all__ = ["parse_pdb"]


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
def parse_pdb(filepath, hetatm=False):
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
    xyz = []
    with open_file(fp) as handle:
        for ln, line in enumerate(handle, start=1):
            if line.startswith(rectypes):
                name = line[12:16]
                element = line[76:78].strip()
                if element == "H":
                    continue
                elif not element and _is_hydrogen(name):
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
                    atom = _Atom(chain_id, resid_num, icode, name)
                    atoms.append(atom)
                    xyz.append((x, y, z))

            elif line.startswith(("MODEL ", "ENDMDL")):
                msg = "Multi-model structures are not supported."
                raise StructureParserException(msg)

    return Structure(filepath, atoms, np.asarray(xyz, dtype=np.float64))
