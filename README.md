[![CI](https://github.com/JoaoRodrigues/fcc/actions/workflows/ci.yml/badge.svg?branch=fcc2)](https://github.com/JoaoRodrigues/fcc/actions/workflows/ci.yml)

# Contact-based clustering for protein structures

Open-source package that implements a fast clustering algorithms
for protein structures based on a contact-based similarity metrics.

Advantages over traditional RMSD-based clustering methods include:
- Does not require alignments.
- As a result, it is usually much faster.
- Multi-resolution clustering: atom-atom contacts, residue-residue, chain-chain, ...
- Works for any molecule you can encode in a PDB/mmCIF format.

## Citation

Rodrigues JPGLM, Trellet M, Schmitz C, Kastritis P, Karaca E, Melquiond ASJ, Bonvin AMJJ.
[Clustering biomolecular complexes by residue contacts similarity.] [1]
Proteins: Structure, Function, and Bioinformatics 2012;80(7):1810–1817.

## Installation

TBD (pip install fccpy)

## Usage & Examples

You can use fccpy as a library or as a set of command-line utilities.

### As a script
```bash
fccpy mkcontacts ppi_complexes.txt
fccpy mtx -m mm ppi_complexes.txt -o similarity.h5
fccpy cluster similarity.h5 dbscan -o clusters.out
```

### As a library
```python
import pathlib
from fccpy import read_pdb, get_intermolecular_contacts

p = pathlib.Path("tests", "inputs", "1brs.pdb.gz")
structure = read_pdb(p)

for atom_a, atom_b in get_intermolecular_contacts(structure):
    print(f"{atom_a}\t{atom_b}")
```

## License

Apache License 2.0 (see [LICENSE](LICENSE) file for details)
