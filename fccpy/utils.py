"""Assorted utility functions."""

import gzip


def open_file(filepath):
    """Return a file-handle from a given file path.

    Infers if file is compressed or not based on extension.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to file to open. If file name ends in 'gz', will
        uncompress the file on the fly.

    Returns
    -------
    file
        an open file handle (text)
    """
    if filepath.suffix == ".gz":
        return gzip.open(filepath, "rt")
    return filepath.open("rt")
