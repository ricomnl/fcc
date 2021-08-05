"""Performance Tests."""

import pytest

from fccc import parse_pdb

pytestmark = [pytest.mark.performance]  # mark all tests as slow


def test_performance_parse_pdb_1K8K(input_dir, benchmark):
    """Parse 1K8K.pdb in a reasonable time: ~10ms"""
    # ~10ms per structure = 100 structures per second
    # 1K8K is a large structure (heptamer)
    benchmark(parse_pdb, (input_dir / "1K8K.pdb"))
    if benchmark.stats.stats.mean > 0.10:
        pytest.fail("Exceeding expected average time of 6ms")
