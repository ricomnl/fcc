"""Assorted utility functions."""

import itertools
import gzip
import math
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


def _chunker(iterable, n):
    """Divides an iterable in generators with n elements each."""
    iterable = iter(iterable)
    while True:
        x = tuple(itertools.islice(iterable, n))
        if not x:  # iterable is empty
            return
        yield x


# Binomial helpers
def get_num_combinations(n):
    """Return the number of pairwise combinations in a set of `n` elements."""
    return (n * (n - 1)) // 2


def get_num_elements(n):
    """Return the number of elements from the number of possible pairwise combinations.

    Essentially, the reverse of the previous function. We only consider the solution
    for [-b + sqrt(b^2 - 4ac)] for practical reasons.
    """
    return int(1 + math.sqrt(1 + (8 * n))) // 2
