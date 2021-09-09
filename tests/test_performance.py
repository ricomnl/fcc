"""Performance Tests."""

import gzip
from pathlib import Path
import shutil
import tempfile

import pytest

from fccpy import read_pdb

pytestmark = [pytest.mark.performance]  # mark all tests as slow


def decompress_file(filepath, cwd=None):
    """Decompress gz file."""

    if cwd is None:
        cwd = filepath.parent
    out_fp = Path(cwd, filepath.stem)  # 1xxx.pdb.gz -> 1xxx.pdb
    with gzip.open(filepath, "rt") as f_in:
        with out_fp.open("wt") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_fp


@pytest.mark.benchmark(group="io")
def test_performance_parse_pdb_1K8K(input_dir, benchmark):
    """Performance test for read_pdb: 1k8k."""
    # Structure does not have element column, worst case scenario for parsing
    # ~10ms per structure = 100 structures per second
    # 1K8K is a large structure (heptamer)
    fp = Path(input_dir, "1k8k.pdb.gz")
    benchmark.pedantic(
        read_pdb,
        args=[fp],
        iterations=10,
        rounds=10,
        warmup_rounds=5,
    )
    if benchmark.stats.stats.mean > 0.2:
        pytest.fail("Exceeding expected average time of 200ms")


@pytest.mark.benchmark(group="io")
def test_performance_parse_pdb_1BRS(input_dir, benchmark):
    """Performance test for read_pdb: 1brs."""
    fp = Path(input_dir, "1brs.pdb.gz")
    benchmark.pedantic(
        read_pdb,
        args=[fp],
        iterations=20,
        rounds=10,
        warmup_rounds=5,
    )
    if benchmark.stats.stats.mean > 0.06:
        pytest.fail("Exceeding expected average time of 6ms")


@pytest.mark.benchmark(group="io")
def test_performance_parse_pdb_1K8K_uncompressed(input_dir, benchmark):
    """Performance test for read_pdb: 1k8k (flat file)."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        gz_fp = Path(input_dir, "1k8k.pdb.gz")
        fp = decompress_file(gz_fp, cwd=tmp_dir)

        # Now benchmark
        benchmark.pedantic(
            read_pdb,
            args=[fp],
            iterations=20,
            rounds=10,
            warmup_rounds=5,
        )
        if benchmark.stats.stats.mean > 0.15:
            pytest.fail("Exceeding expected average time of 150ms")


@pytest.mark.benchmark(group="io")
def test_performance_parse_pdb_1BRS_uncompressed(input_dir, benchmark):
    """Performance test for read_pdb: 1brs (flat file)."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Decompress gz file before parsing
        gz_fp = Path(input_dir, "1brs.pdb.gz")
        fp = decompress_file(gz_fp, cwd=tmp_dir)

        # Now benchmark
        benchmark.pedantic(
            read_pdb,
            args=[fp],
            iterations=20,
            rounds=10,
            warmup_rounds=5,
        )
        if benchmark.stats.stats.mean > 0.06:
            pytest.fail("Exceeding expected average time of 6ms")
