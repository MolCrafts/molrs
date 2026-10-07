"""TRR, the GROMACS full-precision trajectory format — ``molrs::io::trr``.

The doors are functions of :mod:`molrs.io`
(:func:`~molrs.io.read_trr_trajectory`, :func:`~molrs.io.write_trr_trajectory`);
this module holds the format's lazy reader, :class:`TrrReader`.
"""

from .._native import TrrReader

__all__ = ["TrrReader"]
