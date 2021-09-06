#!/usr/bin/env python

"""Parallel calculation of atomic contacts for structures."""

import argparse
from datetime import datetime
from functools import partial
import multiprocessing
import os
from pathlib import Path
import sys
import time

from fccpy import parse_pdb
from fccpy.contacts import get_intermolecular_contacts, write_contacts


# Worker function for multiprocessing
def calculate_contacts(filepath, dmax):
    """Calculate contacts for a single structure."""
    _start_time = time.time_ns()
    s = parse_pdb(filepath)
    clist = get_intermolecular_contacts(s, dmax)

    out_fn = Path(filepath.parent, filepath.stem + ".contacts")
    write_contacts(clist, out_fn)

    _end_time = time.time_ns()
    return (_end_time - _start_time) / 1e9  # return in seconds


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


def validate_num_cpu(arg):
    """Validate the option for -p."""

    max_cpu = multiprocessing.cpu_count()
    try:
        arg = int(arg)
    except ValueError:
        pass
    else:
        if 0 < arg <= max_cpu:
            return arg

    emsg = f"-p must be an integer between 0 and {max_cpu}"
    raise argparse.ArgumentTypeError(emsg)


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
            # Paths are relative to the file where they are written,
            # not relative to this script. So convert them accordingly.
            fp = check_file(Path(filepath.parent, fn.strip()).resolve())
            fp_list.append(fp)
    return fp_list


def get_parser(cmd_args):
    """Create cmd arguments and options parser."""

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "flist",
        type=list_of_paths,
        help=(
            "Input file listing structures to process, one per line. "
            "(default: %(default)s)"
        ),
    )
    ap.add_argument(
        "--max-dist",
        type=is_positive_float,
        default=5.0,
        help=(
            "Distance threshold to define a contact between two atoms. "
            "(default: %(default)s)"
        ),
    )
    ap.add_argument(
        "-p",
        "--num_processes",
        type=validate_num_cpu,
        default=multiprocessing.cpu_count(),
        help=(
            "Number of CPUs to use during calculations. "
            "By default uses all available CPUs, as detected by the "
            "python multiprocessing module. This might not be fully "
            "accurate with queueing systems. For this machine, the "
            "default is %(default)s"
        ),
    )

    # If we are calling from a submodule, the help might be wonky
    # because sys.argv[0] is the script name and cmd_args will be
    # sys.argv[2:]. So, check if sys.argv == cmd_args and if not
    # modify the parser .prog attribute
    if sys.argv != cmd_args:
        ap.prog = f"{os.path.basename(sys.argv[0])} {sys.argv[1]}"

    return ap.parse_args(cmd_args)


def main(cmd_args):
    """Script entry point for setuptools."""

    args = get_parser(cmd_args)

    log("started")
    log(f"command: {' '.join(sys.argv)}")

    _start_time = time.time()
    timings = []
    _worker = partial(calculate_contacts, dmax=args.max_dist)
    with multiprocessing.Pool(args.num_processes) as pool:
        jobs = pool.imap(_worker, args.struct_list)
        for idx, ns in enumerate(jobs, start=1):
            timings.append(ns)

            perc_complete = 100 * idx / len(args.struct_list)
            fp = args.struct_list[idx - 1]
            msg = f" analyzing: {fp.name:<20s} ({perc_complete:>4.1f}%)"
            if idx == len(args.struct_list):
                log(msg)
            else:
                log(msg, end="\r")
    _end_time = time.time()

    cumtime = _end_time - _start_time  # wall clock
    avetime = sum(timings) / len(timings)

    log("finished!")
    log(f"  number of structures processed: {len(args.struct_list)}")
    log(f"  average processing time: {avetime:4.3f} s per structure")
    log(f"  performance: {1/avetime:6.1f} structures per second")
    log(f"  total processing time: {cumtime:4.3f} s")


if __name__ == "__main__":
    main(sys.argv)
