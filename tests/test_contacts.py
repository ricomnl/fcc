"""Code to test contact module."""

from operator import attrgetter
from pathlib import Path
import tempfile

import numpy as np

from fccpy import read_pdb
from fccpy.contacts import get_intermolecular_contacts, get_pairwise_contacts
from fccpy.contacts import hash_contact, hash_many
from fccpy.contacts import read_contacts, write_contacts
from fccpy.structure import Atom
from fccpy.utils import as_file_handle


# Shared variables for tests
atom1 = Atom("A", 1, "", "CA")
atom2 = Atom("A", 1, "", "CB")
atom3 = Atom("B", 2, "", "CB")
p1 = (atom1, atom1)
p2 = (atom1, atom2)
p3 = (atom1, atom3)
pairs = [p1, p2, p3]


# Contact Calculation
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
    s = read_pdb(fp)
    contacts = list(get_intermolecular_contacts(s, 5.0))

    # Read independent data
    ref_fn = Path(input_dir, "1k8k.contacts.gz")
    with as_file_handle(ref_fn, "rt") as handle:
        ref_contacts = []
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
                ref_contacts.append((atom_i, atom_j))

    # Compare to calculated contacts
    assert len(contacts) == len(ref_contacts)

    contacts.sort()
    ref_contacts.sort()

    for (a, b), (ref_a, ref_b) in zip(contacts, ref_contacts):
        assert a.chain == ref_a.chain
        assert a.resid == ref_a.resid
        assert a.name.strip() == ref_a.name.strip()
        assert b.chain == ref_b.chain
        assert b.resid == ref_b.resid
        assert b.name.strip() == ref_b.name.strip()


# Hashing Functions
def test_hash_contact_1():
    """Test hash_contact() without selectors."""

    h1 = hash_contact(p1)
    h2 = hash_contact(p1)
    h3 = hash_contact(p2)

    assert isinstance(h1, str)
    assert h1 == h2
    assert h1 != h3


def test_hash_contact_2():
    """Test hash_contact() with same selectors."""

    by_chain = attrgetter("chain")
    h1 = hash_contact(p1, selector1=by_chain, selector2=by_chain)
    h2 = hash_contact(p2, selector1=by_chain, selector2=by_chain)
    h3 = hash_contact(p3, selector1=by_chain, selector2=by_chain)

    assert isinstance(h1, str)
    assert h1 == h2
    assert h1 != h3


def test_hash_contact_3():
    """Test hash_contact() with different selectors."""

    by_chain = attrgetter("chain")
    h1 = hash_contact(p1, selector1=by_chain)
    h2 = hash_contact(p2, selector1=by_chain)
    h3 = hash_contact(p3, selector1=by_chain)
    h4 = hash_contact(p2, selector2=by_chain)

    assert isinstance(h1, str)
    assert h1 != h2
    assert h1 != h3
    assert h1 != h4


def test_hash_many_1():
    """Test hash_many() with default args."""

    hashes = hash_many(pairs)

    assert isinstance(hashes, frozenset)
    assert len(hashes) == 3

    for p in pairs:
        assert hash_contact(p) in hashes


def test_hash_many_2():
    """Test hash_many() with selector."""

    by_chain = attrgetter("chain")

    pairs = ((atom1, atom1), (atom1, atom2), (atom1, atom3))

    hashes = hash_many(pairs, selector1=by_chain, selector2=by_chain)
    assert len(hashes) == 2

    hashes_ = hash_many(pairs, unique=False, selector1=by_chain, selector2=by_chain)
    assert len(hashes_) == 3
    assert {h.split("_")[0] for h in hashes_} == hashes


# IO
def test_read_write_contacts():
    """Test roundabout read/write contact files."""

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir, "tmp.out")
        write_contacts(pairs, fp)
        pairs_ = list(read_contacts(fp))
        assert len(pairs_) == len(pairs)
        assert set(pairs) == set(pairs_)
