"""XYZ and extended XYZ — ``molrs::io::xyz``.

The doors are functions of :mod:`molrs.io` (:func:`~molrs.io.read_xyz`,
:func:`~molrs.io.read_xyz_trajectory`, :func:`~molrs.io.write_xyz`,
:func:`~molrs.io.write_xyz_trajectory`); this module holds the format's lazy
reader, :class:`XyzReader`.
"""

from .._lib import XyzReader

__all__ = ["XyzReader"]
