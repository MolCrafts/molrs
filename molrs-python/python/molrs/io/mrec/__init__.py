"""Scientific records (``*.mrec``) — ``molrs::io::mrec``.

A record is one self-describing store on disk: ``meta`` plus a snapshot
(``frame``), a topology (``system``), a time-ordered frame sequence
(``trajectory``) and/or a force field (``forcefield``). :class:`~molrs.store.Frame`
and :class:`~molrs.store.Trajectory` are the in-memory objects; every door onto
a store is here.

Whole records, paired like every other format:

* :func:`read` / :func:`write` — Structure (``meta`` + ``frame/``)
* :func:`read_system` / :func:`write_system` — System-def (``meta`` +
  ``system/``)
* :func:`read_trajectory` / :func:`write_trajectory` — Trajectory shape
* :func:`read_forcefield` / :func:`write_forcefield` — force-field package
  (``meta`` + ``forcefield/``; :func:`write` and :func:`write_system` take
  ``forcefield=`` too)
* :func:`read_meta` — the identity document; :func:`section_names` — which
  sections a store holds

A run too large to hold in memory:

* :class:`FrameSequence` — lazy frame cursor over a store (one frame per
  :meth:`FrameSequence.read_frame`; also ``len()``, ``seq[i]``, iteration,
  ``.step`` / ``.time`` labels and ``has_block``)
* :class:`SequenceSchema` / :class:`FrameSequenceWriter` — pin a schema and
  write a run frame by frame

And the rest:

* :class:`ForceFieldSection` — the ``forcefield`` section as data: the
  document and one ``Block`` per style table, units as stored
  (:meth:`molrs.ff.forcefield.ForceField.to_section` / ``from_section`` map
  it onto a force field)
* :func:`pack` — collapse a closed store into one ``*.mrec.zip``
* :mod:`molrs.io.mrec.schema` — runtime check for path suffix and ``meta``
  keys

The names are those of ``molrs::io::mrec`` — ``FrameSequence``,
``FrameSequenceWriter``, ``SequenceSchema``, ``section_names`` — so the lazy
cursor is not confused with :class:`molrs.io.TrajectoryReader`, the
LAMMPS/XYZ/DCD/TRR/XTC dump concatenator.
"""

from ..._lib import mrec as _mrec
from . import schema

ForceFieldSection = _mrec.ForceFieldSection
FrameSequence = _mrec.FrameSequence
FrameSequenceWriter = _mrec.FrameSequenceWriter
SequenceSchema = _mrec.SequenceSchema
pack = _mrec.pack
read = _mrec.read
read_forcefield = _mrec.read_forcefield
read_meta = _mrec.read_meta
read_system = _mrec.read_system
read_trajectory = _mrec.read_trajectory
section_names = _mrec.section_names
write = _mrec.write
write_forcefield = _mrec.write_forcefield
write_system = _mrec.write_system
write_trajectory = _mrec.write_trajectory

__all__ = [
    "ForceFieldSection",
    "FrameSequence",
    "FrameSequenceWriter",
    "SequenceSchema",
    "pack",
    "read",
    "read_forcefield",
    "read_meta",
    "read_system",
    "read_trajectory",
    "schema",
    "section_names",
    "write",
    "write_forcefield",
    "write_system",
    "write_trajectory",
]
