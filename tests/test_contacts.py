"""Code to test contact module."""

from pathlib import Path

import numpy as np

from fccpy import parse_pdb
from fccpy import get_intermolecular_contacts
from fccpy.structure import Atom
from fccpy.contacts import get_pairwise_contacts
from fccpy.utils import open_file


def test_get_pairwise_contacts():
    """Test pairwise contact calculation on arrays."""

    coords = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float64)

    idx_i = np.array([0], dtype=np.int64)
    idx_j = np.array([1], dtype=np.int64)
    i, j = get_pairwise_contacts(coords, idx_i, idx_j, 0.0)
    assert len(i) == len(j) == 0

    i, j = get_pairwise_contacts(coords, idx_i, idx_j, 1.0)
    assert i[0] == 0 and j[0] == 1  # array values, not array indices


def test_get_intermolecular_contacts(input_dir):
    """Test intermolecular contact calculation on 1k8k.pdb."""

    fp = Path(input_dir, "1k8k.pdb.gz")
    s = parse_pdb(fp)
    contacts = list(get_intermolecular_contacts(s, 5.0))

    # Read independent data
    ref_fn = Path(input_dir, "1k8k.contacts.gz")
    with open_file(ref_fn) as handle:
        true_contacts = []
        for lineno, line in enumerate(handle):
            try:
                (
                    resid_i,
                    chain_i,
                    name_i,
                    resid_j,
                    chain_j,
                    name_j,
                    _,
                ) = line.strip().split()
            except Exception:  # noqa: B902
                if line.strip():
                    msg = f"Error parsing line {lineno} of file: {ref_fn.name}"
                    raise IOError(msg)
            else:
                atom_i = Atom(chain_i, int(resid_i), "", name_i)
                atom_j = Atom(chain_j, int(resid_j), "", name_j)
                true_contacts.append((atom_i, atom_j))

    # Compare to calculated contacts
    assert len(contacts) == len(true_contacts)

    contacts.sort()
    true_contacts.sort()

    for c, tc in zip(contacts, true_contacts):
        c_a, c_b = c
        tc_a, tc_b = tc
        # This is annoying because of icodes
        for prop in ["chain", "resid", "name"]:
            assert getattr(c_a, prop) == getattr(tc_a, prop)
            assert getattr(c_b, prop) == getattr(tc_b, prop)
