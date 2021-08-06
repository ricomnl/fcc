"""Main entry point for fccpy CLI.

Implements top-level parser and dispatches arguments to the
relevant submodules.
"""

import argparse
from dataclasses import dataclass
import sys
import textwrap
from typing import Callable

from fccpy.cli_submodules import mkcontacts

__all__ = ["main"]


@dataclass
class Subcommand:
    name: str
    description: str
    func: Callable


AVAILABLE_SUBCOMMANDS = (
    Subcommand(
        name="mkcontacts",
        description="calculates contacts in one or more structures.",
        func=mkcontacts.main,
    ),
)

# Dynamically create usage string
_subcmd_str = "\n".join(
    f"\t{subcmd.name}\t\t{subcmd.description}\n" for subcmd in AVAILABLE_SUBCOMMANDS
)

_usage = textwrap.dedent(
    f"""
fccpy <command> [<args>]

List of available commands:
{_subcmd_str}
    """
)


# Main function
def main():
    """Top-level script for fccpy."""

    ap = argparse.ArgumentParser(
        description=__doc__,
        usage=_usage,
    )

    ap.add_argument(
        "command",
        choices=[s.name for s in AVAILABLE_SUBCOMMANDS],
        help="Sub-command to run",
    )

    # Parse subcommand only
    args = ap.parse_args(sys.argv[1:2])

    # Get SubCommand and invoke function
    # Silly, but this costs peanuts
    for subcmd in AVAILABLE_SUBCOMMANDS:
        if args.command == subcmd.name:
            break
    else:
        raise Exception("Er... this should not happen..")

    subcmd.func(sys.argv[2:])


if __name__ == "__main__":
    main()
