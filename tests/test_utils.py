"""Tests for utility functions."""

from fccc.structure import parse_pdb, divide_by_chain


def test_divide_by_chain(input_dir):
    """Test divide_by_chain()"""

    expected_chains = ["A", "B", "C", "D", "E", "F", "G"]
    expected_atoms = [3214, 1517, 2647, 2293, 1416, 1371, 872]

    structure = parse_pdb(input_dir / "1K8K.pdb")
    chains = divide_by_chain(structure)
    assert len(chains) == 7
    assert list(chains) == expected_chains

    for chain, n_atoms in zip(expected_chains, expected_atoms):
        chain_atoms = sum(1 for a in structure if a.chain == chain)
        assert chain_atoms == n_atoms
