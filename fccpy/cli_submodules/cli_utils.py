"""Utilitity Functions for CLI submodules."""

import argparse
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path


def log(msg, **kwargs):
    """Write a message to stdout with a timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", **kwargs)


def is_positive_float(n):
    """Return n if n is positive and non-zero."""
    try:
        n = float(n)
    except ValueError:
        msg = f"{n!r} ({type(n)=}) cannot be converted to a float."
        raise argparse.ArgumentTypeError(msg)
    else:
        if n > 0:
            return n
    raise argparse.ArgumentTypeError(f"{n!r} must be positive and non-zero.")


def validate_num_cpu(arg):
    """Validate the option for -p."""

    max_cpu = cpu_count()
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
