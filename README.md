[![CI](https://github.com/JoaoRodrigues/fcc/actions/workflows/ci.yml/badge.svg?branch=fcc2)](https://github.com/JoaoRodrigues/fcc/actions/workflows/ci.yml)

Contact-based clustering for protein structures
============================================

Open-source package that implements a fast clustering algorithm for protein
structures based on a contact-based similarity metric.

Advantages over traditional RMSD-based clustering methods include:
- Does not require alignments.
- As a result, it is usually much faster.
- Multi-resolution clustering: atom-atom contacts, residue-residue, chain-chain, ...
- Works for any molecule you can encode in a PDB/mmCIF format.

Citation
========

Rodrigues JPGLM, Trellet M, Schmitz C, Kastritis P, Karaca E, Melquiond ASJ, Bonvin AMJJ.
[Clustering biomolecular complexes by residue contacts similarity.] [1]
Proteins: Structure, Function, and Bioinformatics 2012;80(7):1810–1817.

Installation
===========

TBD (pip install fccc)

License
=======

TBD
