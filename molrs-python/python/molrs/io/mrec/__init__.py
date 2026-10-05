"""Scientific-record I/O for ``*.mrec`` stores — the store machinery.

``Frame`` and ``Trajectory`` are the in-memory objects. The whole-record path
doors sit with every other format in :mod:`molrs.io`, paired like the rest:
:func:`~molrs.io.read_mrec` / :func:`~molrs.io.write_mrec` (Structure,
``meta`` + ``frame/``), :func:`~molrs.io.read_mrec_system` /
:func:`~molrs.io.write_mrec_system` (System-def, ``meta`` + ``system/``) and
:func:`~molrs.io.read_mrec_trajectory` /
:func:`~molrs.io.write_mrec_trajectory` (Trajectory shape) and
:func:`~molrs.io.read_mrec_forcefield` /
:func:`~molrs.io.write_mrec_forcefield` (force-field package, ``meta`` +
``forcefield/``; ``write_mrec`` / ``write_mrec_system`` take ``forcefield=``
too), with
:func:`~molrs.io.read_mrec_meta` for the identity document and
:func:`~molrs.io.mrec_sections` to ask which sections a store holds. What only
a store has lives here:

* :class:`TrajectoryReader` — lazy frame cursor over Rust ``FrameSequence``
  (one frame per :meth:`TrajectoryReader.read_frame`; also ``len()``,
  ``reader[i]``, iteration, ``.step`` / ``.time`` labels and ``has_block``)
* :class:`SequenceSchema` / :class:`TrajectoryWriter` — pin a schema and write
  a run frame by frame, without holding it all in memory
* :class:`ForceFieldSection` — the ``forcefield`` section as data: the
  document and one ``Block`` per style table, units as stored
  (:meth:`molrs.ff.ForceField.to_section` / ``from_section`` map it onto a
  force field)
* :func:`pack` — collapse a closed store into one ``*.mrec.zip``
* :mod:`molrs.io.mrec.schema` — runtime check for path suffix and ``meta`` keys

These names are not re-exported from :mod:`molrs.io`. That module's
:class:`~molrs.io.TrajectoryReader` is the LAMMPS/XYZ/DCD dump concatenator.
"""

from ..._lib import (
    ForceFieldSection,
    SequenceSchema,
    TrajectoryReader,
    TrajectoryWriter,
    pack,
)
from . import schema

__all__ = [
    "ForceFieldSection",
    "SequenceSchema",
    "TrajectoryReader",
    "TrajectoryWriter",
    "pack",
    "schema",
]
