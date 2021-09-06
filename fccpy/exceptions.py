"""Exception classes."""


class StructureParserException(Exception):
    """Exception to signal errors when parsing PDB/mmCIF files."""

    pass


class MatrixReadingException(Exception):
    """Exception to signal error when reading matrices from file."""
