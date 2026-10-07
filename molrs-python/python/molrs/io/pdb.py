"""PDB, the Protein Data Bank coordinate format — ``molrs::io::pdb``.

The doors are functions of :mod:`molrs.io` (:func:`~molrs.io.read_pdb`,
:func:`~molrs.io.read_pdb_trajectory`, :func:`~molrs.io.write_pdb`,
:func:`~molrs.io.write_pdb_trajectory`); this module holds the format's
lazy reader, :class:`PdbReader` (every ``MODEL`` of one or several files).
"""

from .._lib import PdbReader

__all__ = ["PdbReader"]
