"""DCD, the CHARMM / NAMD binary trajectory format — ``molrs::io::dcd``.

The doors are functions of :mod:`molrs.io`
(:func:`~molrs.io.read_dcd_trajectory`, :func:`~molrs.io.write_dcd_trajectory`);
this module holds the format's lazy reader, :class:`DcdReader` (O(1) random
access).
"""

from .._lib import DcdReader

__all__ = ["DcdReader"]
