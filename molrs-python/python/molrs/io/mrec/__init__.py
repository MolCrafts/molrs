"""Scientific records (``*.mrec``) — ``molrs::io::mrec``.

A record is one self-describing store on disk: ``meta`` plus a snapshot
(``frame``), a topology (``system``), a time-ordered frame sequence
(``trajectory``) and/or a force field (``forcefield``).
:class:`~molrs.core.Frame` and :class:`~molrs.core.Trajectory` are the
in-memory objects.

Whole records are read and written like every other format, by functions at
the top of :mod:`molrs.io`: :func:`~molrs.io.read_mrec_frame` /
:func:`~molrs.io.write_mrec_frame` (Structure: ``meta`` + ``frame/``), their
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
  (:meth:`ForceFieldSection.from_forcefield` /
  :meth:`ForceFieldSection.to_forcefield` map it onto a force field)
* :func:`section_names` — which sections a store holds
* :func:`pack_mrec_zip` — collapse a closed store into one ``*.mrec.zip``
* :data:`MOLREC_VERSION` / :data:`RESERVED_META_KEYS` — the record
  contract's version and the ``meta`` keys it reserves
* :mod:`molrs.io.mrec.validation` — runtime check for path suffix and
  ``meta`` keys

The names are those of ``molrs::io::mrec`` — ``MrecReader``, ``MrecWriter``,
``SequenceSchema``, ``section_names``.
"""

from ..._lib import mrec as _mrec
from . import validation

ForceFieldSection = _mrec.ForceFieldSection
MOLREC_VERSION = _mrec.MOLREC_VERSION
RESERVED_META_KEYS = _mrec.RESERVED_META_KEYS
MrecReader = _mrec.MrecReader
MrecWriter = _mrec.MrecWriter
SequenceSchema = _mrec.SequenceSchema
pack_mrec_zip = _mrec.pack_mrec_zip
section_names = _mrec.section_names

__all__ = [
    "MOLREC_VERSION",
    "RESERVED_META_KEYS",
    "ForceFieldSection",
    "MrecReader",
    "MrecWriter",
    "SequenceSchema",
    "pack_mrec_zip",
    "section_names",
    "validation",
]
