"""Main entry point for fccpy CLI.

Implements top-level parser and dispatches arguments to the
relevant submodules.

To add support for a new subcommand:
    - Place the file in `cli_submodules`
    - Create a new Subcommand instance under `SUBCOMMANDS`
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


SUBCOMMANDS = (
    Subcommand(
        name="mkcontacts",
        description="calculates contacts in one or more structures.",
        func=mkcontacts.main,
    ),
)

# Dynamically create usage string
_subcmd_str = "\n".join(
    f"  {subcmd.name}\t\t{subcmd.description}\n" for subcmd in SUBCOMMANDS
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
        usage=_usage,
    )

    ap.add_argument(
        "command",
        help="Sub-command to run",
    )

    # Parse subcommand only
    args = ap.parse_args(sys.argv[1:2])

    # Get SubCommand and invoke function
    # Silly, but this costs peanuts
    for subcmd in SUBCOMMANDS:
        if args.command == subcmd.name:
            break
    else:
        raise ValueError(f"Unknown subcommand: {args.command!r}")

    subcmd.func(sys.argv[2:])


if __name__ == "__main__":
    main()
