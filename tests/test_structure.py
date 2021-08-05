"""Code to test structure module classes and functions."""

from fccc import parse_pdb
from fccc.structure import divide_by_chain


def test_parse_pdb_1k8k(input_dir):
    """Parse 1k8k.pdb"""

    expected_chains = ["A", "B", "C", "D", "E", "F", "G"]
    expected_atoms = [3214, 1517, 2647, 2293, 1416, 1371, 872]

    structure = parse_pdb(input_dir / "1k8k.pdb.gz")

    assert len(structure.atoms) == 13330  # heavy atoms only
    assert sorted({a.chain for a in structure}) == expected_chains

    for chain, n_atoms in zip(expected_chains, expected_atoms):
        chain_atoms = sum(1 for a in structure if a.chain == chain)
        assert chain_atoms == n_atoms


def test_parse_pdb_1BRS(input_dir):
    """Parse 1brs.pdb"""

    expected_chains = ["A", "D"]
    expected_atoms = [864, 695]

    structure = parse_pdb(input_dir / "1brs.pdb.gz")

    assert len(structure.atoms) == sum(expected_atoms)
    assert sorted({a.chain for a in structure}) == expected_chains

    for chain, n_atoms in zip(expected_chains, expected_atoms):
        chain_atoms = sum(1 for a in structure if a.chain == chain)
        assert chain_atoms == n_atoms


def test_parse_pdb_1BRS_w_hetatm(input_dir):
    """Parse 1brs.pdb with HETATM."""

    expected_chains = ["A", "D"]
    expected_atoms_w_het = [1009, 772]

    structure = parse_pdb(input_dir / "1brs.pdb.gz", hetatm=True)

    assert len(structure.atoms) == sum(expected_atoms_w_het)
    assert sorted({a.chain for a in structure}) == expected_chains

    for chain, n_atoms in zip(expected_chains, expected_atoms_w_het):
        chain_atoms = sum(1 for a in structure if a.chain == chain)
        assert chain_atoms == n_atoms


def test_divide_by_chain(input_dir):
    """Test divide_by_chain()"""

    expected_chains = ["A", "B", "C", "D", "E", "F", "G"]
    expected_atoms = [3214, 1517, 2647, 2293, 1416, 1371, 872]

    structure = parse_pdb(input_dir / "1k8k.pdb.gz")
    chains = divide_by_chain(structure)
    assert len(chains) == 7
    assert list(chains) == expected_chains

    for chain, n_atoms in zip(expected_chains, expected_atoms):
        chain_atoms = sum(1 for a in structure if a.chain == chain)
        assert chain_atoms == n_atoms