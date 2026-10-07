"""XTC, the GROMACS compressed trajectory format — ``molrs::io::xtc``.

The doors are functions of :mod:`molrs.io`
(:func:`~molrs.io.read_xtc_trajectory`, :func:`~molrs.io.write_xtc_trajectory`);
this module holds the format's lazy reader, :class:`XtcReader`.
"""

from .._native import XtcReader

__all__ = ["XtcReader"]
