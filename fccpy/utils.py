"""Assorted utility functions."""

import gzip


def open_file(filepath):
    """Return a file-handle from a given file path.

    Infers if file is compressed or not based on extension.
    """
    if filepath.suffix == ".gz":
        return gzip.open(filepath, "rt")
    return filepath.open("rt")
