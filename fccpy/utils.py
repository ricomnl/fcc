"""Assorted utility functions."""

import gzip
from pathlib import Path


def as_file_handle(filepath, mode):
    """Return a file-handle from a given file path.

    Infers if file is compressed or not based on extension.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to file to open. If file name ends in 'gz', will
        uncompress the file on the fly.
    mode : any supported mode for file or GzipFile

    Returns
    -------
    file
        an open file handle (text)
    """
    filepath = Path(filepath)
    if filepath.suffix == ".gz":
        return gzip.open(filepath, mode)
    return filepath.open(mode)
