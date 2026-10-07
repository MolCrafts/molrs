"""GRO, the GROMACS structure / trajectory format — ``molrs::io::gro``.

The doors are functions of :mod:`molrs.io` (:func:`~molrs.io.read_gro`,
:func:`~molrs.io.read_gro_trajectory`, :func:`~molrs.io.write_gro`,
:func:`~molrs.io.write_gro_trajectory`); this module holds the format's lazy
reader, :class:`GroReader`.
"""

from .._lib import GroReader

__all__ = ["GroReader"]
