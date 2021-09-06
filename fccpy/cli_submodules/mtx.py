#!/usr/bin/env python

"""Calculate a similarity matrix based on contacts.

Users can select from a variety of pre-set similarity metrics
and modes of comparison, adequate for different biological systems.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import textwrap
import time

from fccpy.contacts import BY_ATOM, BY_RESIDUE, BY_RESIDUE_NOCHAIN
from fccpy.contacts import read_contacts, hash_many
from fccpy.similarity import build_matrix, write_matrix
from fccpy.similarity import jaccard, overlap, fcc  # metrics

# Update here if necessary
METRICS = {"jaccard": jaccard, "overlap": overlap, "fcc": fcc}


# Utilitity Functions
def log(msg, **kwargs):
    """Write a message to stdout with a timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", **kwargs)


def is_positive_float(n):
    """Return n if n is positive and non-zero."""
    assert isinstance(n, float), f"{n!r} ({type(n)=}) must be of type float."
    if n > 0:
        return n
    raise argparse.ArgumentTypeError(f"{n!r} must be positive and non-zero.")


def check_file(fn):
    """Validate that the file exists and is readable."""
    fp = Path(fn)
    if not fp.exists():
        raise argparse.ArgumentTypeError(f"{fp!r} does not exist.")
    elif not fp.is_file():
        raise argparse.ArgumentTypeError(f"{fp!r} is not a file.")
    return fp


def list_of_paths(filepath):
    """Parse a file containing one file path per line."""
    filepath = Path(filepath)
    fp_list = []
    with filepath.open("rt") as handle:
        for fn in handle:
            structure_fp = Path(filepath.parent, fn.strip()).resolve()
            contact_fp = structure_fp.with_suffix(".contacts")
            # Paths are relative to the file where they are written,
            # not relative to this script. So convert them accordingly.
            fp = check_file(contact_fp)
            fp_list.append(fp)
    return fp_list


def get_parser(cmd_args):
    """Create cmd arguments and options parser."""

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "flist",
        type=list_of_paths,
        help=(
            "Input file listing structures to process, one per line. "
            "Code will automatically find corresponding contact files "
            "produced by mkcontacts. "
        ),
    )
    ap.add_argument(
        "-o",
        "--output_file",
        type=Path,
        default="similarity.h5",
        help=(
            "Output file name to write matrix to. Matrix is stored in "
            "HDF5 format, as a matrix of floats. (default: %(default)s)"
        ),
    )
    ap.add_argument(
        "-s",
        "--similarity",
        dest="metric",
        choices=METRICS.keys(),
        default="fcc",
        help=(
            "Similarity metric to use when comparing structures. "
            "(default: %(default)s)"
        ),
    )
    ap.add_argument(
        "-m",
        "--mode",
        choices=["mm", "lig", "sym"],
        required=True,
        help=textwrap.dedent(
            """
            Different modes of comparing pairs of atoms:

            mm: for macromolecular complexes. Will compare contacts
            based on residue information, ignoring atom information.

            lig: will consider residue information for the first atom in
            each contact, and full atom information for the second.
            Expects (but does not check!) that the contact files only have
            information for two chains.

            sym: for symmetrical homo-multimers, will compare residue
            information for each pair of atoms in contact, ignoring the chain
            identifier.

            (default: %(default)s)
            """
        ),
    )

    # If we are calling from a submodule, the help might be wonky
    # because sys.argv[0] is the script name and cmd_args will be
    # sys.argv[2:]. So, check if sys.argv == cmd_args and if not
    # modify the parser .prog attribute
    if sys.argv != cmd_args:
        ap.prog = f"{os.path.basename(sys.argv[0])} {sys.argv[1]}"
    else:
        cmd_args = sys.argv[1:]  # remove script name

    return ap.parse_args(cmd_args)


def main(cmd_args):
    """Script entry point for setuptools."""

    args = get_parser(cmd_args)

    log("started")
    log(f"command: {' '.join(sys.argv)}")

    # Check which mode of comparison to use and make appropriate choices.
    if args.mode == "mm":
        selector1 = selector2 = BY_RESIDUE
        unique = True
    elif args.mode == "sym":
        selector1 = selector2 = BY_RESIDUE_NOCHAIN
        unique = False
    elif args.mode == "lig":
        selector1 = BY_RESIDUE
        selector2 = BY_ATOM
        unique = True

    _start_time1 = time.time()

    clist = [list(read_contacts(f)) for f in args.flist]
    _end_time = time.time()
    read_time = _end_time - _start_time1
    log(f"  read {len(clist)} files in {read_time:4.2f} s")

    # Make contact hashes
    _start_time2 = time.time()
    clist_hashed = [
        hash_many(c, unique=unique, selector1=selector1, selector2=selector2)
        for c in clist
    ]
    _end_time = time.time()
    hash_time = _end_time - _start_time2
    n_total_hashes = sum(map(len, clist_hashed))
    log(f"  hashed {n_total_hashes} pairs in {hash_time:4.2f} s")

    # Build matrix
    _start_time3 = time.time()
    idxs, sims = build_matrix(clist_hashed, metric=METRICS[args.metric])
    _end_time = time.time()
    mtx_time = _end_time - _start_time3

    n_total_comp = (len(clist) ** 2 // 2) - (len(clist) // 2)
    log(f"  built similarity matrix ({n_total_comp} comparisons) in {mtx_time:4.2f} s")

    # Write to file
    log(f"  writing matrix to file: {args.output_file}")
    write_matrix(idxs, sims, args.output_file)

    _end_time = time.time()
    cumtime = _end_time - _start_time1

    # statistics
    log("finished!")
    log(f"  total processing time: {cumtime:4.2f} s")


if __name__ == "__main__":
    main(sys.argv)
