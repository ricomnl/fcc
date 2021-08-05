"""Config file for pytest."""

import pathlib

import pytest


# Define data folder relative to this file's location
@pytest.fixture(scope="session")
def input_dir():
    """Define data folder."""
    this = pathlib.Path(__file__).resolve().parent
    return this / "inputs"


# Skip performance tests by default
def pytest_addoption(parser):
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="run performance tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance (slow to run)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-performance"):
        # --run-performance given in cli: do not skip performance tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-performance option to run")
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_slow)
