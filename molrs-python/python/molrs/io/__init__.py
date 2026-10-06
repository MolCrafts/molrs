"""File I/O — ``molrs::io``.

Structure and trajectory file formats, SMILES and CGsmiles text, and
``*.mrec`` records. Every reader emits the project-wide canonical column names
(:mod:`molrs.store.keys`: ``element``, ``res_id``, ``charge``, ``mol_id``,
…) — the Rust readers map a format's own spelling (``symbol``, ``resSeq``,
``q``, ``mol``) at the boundary — and every writer takes them. Force-field
file formats are :mod:`molrs.ff.forcefield`'s.

Names come in pairs. A format ``X`` that holds one frame is read by
``read_X`` and written by ``write_X``; a sequence of frames is read by
``read_X_trajectory`` and written by ``write_X_trajectory`` — ``lammps`` dumps,
``xyz``, ``pdb``, ``gro``, ``dcd``, ``trr`` and ``xtc``. A format
with one direction only (``read_chgcar``, ``write_lammps_dump_local``, …) has
no partner by nature, not by omission.

``read_lammps_trajectory``, ``read_xyz_trajectory``, ``read_dcd_trajectory``,
``read_trr_trajectory`` and ``read_xtc_trajectory`` return a lazy
:class:`TrajectoryReader` rather than a ``list[Frame]``; each accepts a single
path or a list of paths (frames are concatenated) and yields canonical field
names. ``read_pdb_trajectory`` and ``read_gro_trajectory`` read the whole file
into a ``list[Frame]`` (no seekable native reader backs them). Every path
argument takes a ``str`` or any ``os.PathLike``.

A ``*.mrec`` scientific record is a store, not a file format: every door onto
one — whole-record ``read`` / ``write`` and their ``system`` / ``trajectory`` /
``forcefield`` partners, the lazy :class:`~molrs.io.mrec.FrameSequence`
cursor and its writer — is :mod:`molrs.io.mrec`'s.

:class:`SmilesIR` is here because SMILES is a *format*: text in, molecule out,
exactly like PDB or XYZ. SMARTS is not — a pattern is a query over a perceived
graph — so it lives in :mod:`molrs.perceive`.

:class:`CGSmilesIR` is the same kind of door onto the CGsmiles notation, which
writes a molecule at one or more *coarse-grained* resolutions — a resolution
at which one particle, a *bead*, stands in for a whole group of atoms: text
in, one :class:`CGGraph` per resolution level plus the fragment tables that
resolve them out, and ``to_atomistic()`` expands the lowest level into atoms.
That expansion is topology only — atoms, bonds and the per-atom ``frag_id``
saying which bead each atom came from. A line notation states no geometry, so
coordinates, hydrogens and perception remain separate steps. The records it
hands out — :class:`CGGraph`, :class:`CGNode`, :class:`CGEdge`,
:class:`CGFragmentDef`, :class:`ResolvedPair`, :class:`PairEnd` and
:class:`BondingDescriptor` — are read-only views over the parsed value, so no
fact of the notation has to be re-parsed, decoded or unpacked from a bare
tuple position on the Python side.

A :class:`BondingDescriptor` reports its ``kind`` as the grammar glyph
(``"$"``, ``"<"``, ``">"``, ``"!"``), which is both what a user writes and
what a stored port's ``port_kind`` prop holds — one spelling for the notation,
the column and this boundary, so a descriptor kind reaches
:meth:`Atomistic.def_port <molrs.system.Atomistic.def_port>` untranslated. The enums
the notation does not spell out keep lowercase variant names:
``BondingDescriptor.order`` and ``ResolvedPair.kind`` are bond kinds
(``"single"``, ``"aromatic"``, …) and ``PairEnd.end`` is ``"sub"`` or
``"body"``.

Every refusal raised by this family of notations — by the parser, by the
expansion of a CGsmiles string, or by an emit — is a :class:`SmilesError`, one
class carrying the four facts the Rust error owns: ``kind``, the variant name
of the rule that was broken (``"UnclosedBranch"``, ``"UnexpectedEnd"``,
``"CgNotExpandable"``, …); ``span``, the byte range of the offending text as a
``(start, end)`` pair whose end is clamped to ``len(input)``; ``input``, the
offending string, empty for the errors raised past the parser, which are handed
an IR and never see the text it came from; and ``notation``, lowercase
``"smiles"``, ``"smarts"`` or ``"cgsmiles"``. It subclasses
:class:`ValueError`, so ``except ValueError`` keeps catching it, and
``str(e)`` is the message Rust renders, caret line included.

There is no ``CGSmilesReader``, deliberately: "Reader" in this module means a
lazy, path-backed trajectory cursor (:class:`TrajectoryReader`), and a
text-in / IR-out parser is not that object. A reader-shaped ``read()`` API over CGsmiles belongs to
molpy, which wraps :class:`CGSmilesIR` exactly as its ``SmilesReader`` wraps
:class:`SmilesIR`.
"""

from .._lib import (
    BondReactTemplate,
    BondingDescriptor,
    CGEdge,
    CGFragmentDef,
    CGGraph,
    CGNode,
    CGSmilesIR,
    LammpsCpuUse,
    LammpsLoadBalance,
    LammpsLog,
    LammpsLogHeader,
    LammpsLoopTime,
    LammpsMemoryUsage,
    LammpsNeighborStatistics,
    LammpsPerformance,
    LammpsRun,
    LammpsThermo,
    LammpsTimingBreakdown,
    LammpsTimingRow,
    LammpsWarning,
    PairEnd,
    ResolvedPair,
    SmilesError,
    SmilesIR,
    parse_lammps_log_text,
    read_ac,
    read_amber_inpcrd,
    read_amber_prmtop,
    read_chgcar,
    read_cube,
    read_frame,
    read_gro,
    read_gro_trajectory,
    read_lammps_data,
    read_lammps_log,
    read_lammps_molecule,
    read_mol2,
    read_pdb,
    read_pdb_trajectory,
    read_prep,
    read_stl,
    read_xsf,
    read_xyz,
    write_bond_react_map,
    write_cube,
    write_dcd_trajectory,
    write_frame,
    write_gro,
    write_gro_trajectory,
    write_lammps_bond_react_system,
    write_lammps_data,
    write_lammps_dump_local,
    write_lammps_molecule,
    write_lammps_trajectory,
    write_mol2,
    write_pdb,
    write_pdb_trajectory,
    write_prep,
    write_smarts,
    write_trr_trajectory,
    write_xsf,
    write_xtc_trajectory,
    write_xyz,
    write_xyz_trajectory,
)
from . import mrec
from ._csv import read_block_csv, write_block_csv
from ._trajectory import (
    TrajectoryReader,
    read_dcd_trajectory,
    read_lammps_trajectory,
    read_trr_trajectory,
    read_xtc_trajectory,
    read_xyz_trajectory,
)

# Defined in the private modules above; this module is their public path.
for _defined in (
    TrajectoryReader,
    read_block_csv,
    read_dcd_trajectory,
    read_lammps_trajectory,
    read_trr_trajectory,
    read_xtc_trajectory,
    read_xyz_trajectory,
    write_block_csv,
):
    _defined.__module__ = __name__

__all__ = [
    "BondReactTemplate",
    "BondingDescriptor",
    "CGEdge",
    "CGFragmentDef",
    "CGGraph",
    "CGNode",
    "CGSmilesIR",
    "LammpsCpuUse",
    "LammpsLoadBalance",
    "LammpsLog",
    "LammpsLogHeader",
    "LammpsLoopTime",
    "LammpsMemoryUsage",
    "LammpsNeighborStatistics",
    "LammpsPerformance",
    "LammpsRun",
    "LammpsThermo",
    "LammpsTimingBreakdown",
    "LammpsTimingRow",
    "LammpsWarning",
    "PairEnd",
    "ResolvedPair",
    "SmilesError",
    "SmilesIR",
    "TrajectoryReader",
    "mrec",
    "parse_lammps_log_text",
    "read_ac",
    "read_amber_inpcrd",
    "read_amber_prmtop",
    "read_block_csv",
    "read_chgcar",
    "read_cube",
    "read_dcd_trajectory",
    "read_frame",
    "read_gro",
    "read_gro_trajectory",
    "read_lammps_data",
    "read_lammps_log",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    "read_mol2",
    "read_pdb",
    "read_pdb_trajectory",
    "read_prep",
    "read_stl",
    "read_trr_trajectory",
    "read_xsf",
    "read_xtc_trajectory",
    "read_xyz",
    "read_xyz_trajectory",
    "write_block_csv",
    "write_cube",
    "write_dcd_trajectory",
    "write_bond_react_map",
    "write_frame",
    "write_gro",
    "write_gro_trajectory",
    "write_lammps_bond_react_system",
    "write_lammps_data",
    "write_lammps_dump_local",
    "write_lammps_molecule",
    "write_lammps_trajectory",
    "write_mol2",
    "write_pdb",
    "write_pdb_trajectory",
    "write_prep",
    "write_smarts",
    "write_trr_trajectory",
    "write_xsf",
    "write_xtc_trajectory",
    "write_xyz",
    "write_xyz_trajectory",
]
