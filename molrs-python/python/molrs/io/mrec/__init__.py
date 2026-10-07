"""Scientific records (``*.mrec``) — ``molrs::io::mrec``.

A record is one self-describing store on disk: ``meta`` plus a snapshot
(``frame``), a topology (``system``), a time-ordered frame sequence
(``trajectory``) and/or a force field (``forcefield``).
:class:`~molrs.store.Frame` and :class:`~molrs.store.Trajectory` are the
in-memory objects.

Whole records are read and written like every other format, by functions at
the top of :mod:`molrs.io`: :func:`~molrs.io.read_mrec` /
:func:`~molrs.io.write_mrec` (Structure: ``meta`` + ``frame/``), their
``_system`` / ``_trajectory`` / ``_forcefield`` partners, and
:func:`~molrs.io.read_mrec_meta` (the identity document).

This module holds the rest:

* :class:`MrecReader` — lazy frame cursor over a store (one frame per
  :meth:`MrecReader.read_frame`; also ``len()``, ``seq[i]``, iteration,
  ``.step`` / ``.time`` labels and ``has_block``)
* :class:`SequenceSchema` / :class:`MrecWriter` — pin a schema and write a run
  frame by frame
* :class:`ForceFieldSection` — the ``forcefield`` section as data: the
  document and one ``Block`` per style table, units as stored
  (:meth:`molrs.ff.forcefield.ForceField.to_section` / ``from_section`` map
  it onto a force field)
* :func:`section_names` — which sections a store holds
* :func:`pack` — collapse a closed store into one ``*.mrec.zip``
* :mod:`molrs.io.mrec.schema` — runtime check for path suffix and ``meta``
  keys

The names are those of ``molrs::io::mrec`` — ``MrecReader``, ``MrecWriter``,
``SequenceSchema``, ``section_names``.
"""

from ..._lib import mrec as _mrec
from . import schema

ForceFieldSection = _mrec.ForceFieldSection
MrecReader = _mrec.MrecReader
MrecWriter = _mrec.MrecWriter
SequenceSchema = _mrec.SequenceSchema
pack = _mrec.pack
section_names = _mrec.section_names

__all__ = [
    "ForceFieldSection",
    "MrecReader",
    "MrecWriter",
    "SequenceSchema",
    "pack",
    "schema",
    "section_names",
]
