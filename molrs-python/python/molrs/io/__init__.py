"""File I/O — ``molrs::io``.

Every reader here applies a :class:`FieldFormatter <molrs.fields.FieldFormatter>`
to translate format-native column names (``resid``, ``q``, ``symbol``) into
project-wide canonical names (``res_id``, ``charge``, ``element``). Writers
apply the reverse translation before delegating to the native backend. That
canonicalization is the whole point of this layer, so it is where callers
should start.

The un-translated bindings live in :mod:`molrs.io.raw`, one-to-one with the
Rust ``molrs::io`` surface. Use those when the format-native spelling *is* the
thing under test.

Names come in pairs. A format ``X`` that holds one frame is read by
``read_X`` and written by ``write_X``; a sequence of frames is read by
``read_X_trajectory`` and written by ``write_X_trajectory`` — ``lammps`` dumps,
``xyz``, ``pdb``, ``gro``, ``dcd``, ``trr``, ``xtc`` and ``mrec``. A format
with one direction only (``read_chgcar``, ``write_lammps_dump_local``, …) has
no partner by nature, not by omission.

``read_lammps_trajectory``, ``read_xyz_trajectory``, ``read_dcd_trajectory``,
``read_trr_trajectory`` and ``read_xtc_trajectory`` return a lazy
:class:`TrajectoryReader` rather than a ``list[Frame]``; each accepts a single
path or a list of paths (frames are concatenated) and yields canonical field
names. ``read_pdb_trajectory`` and ``read_gro_trajectory`` read the whole file
into a ``list[Frame]`` (no seekable native reader backs them), and
``read_mrec_trajectory`` returns the in-memory :class:`~molrs.Trajectory`;
:class:`molrs.io.mrec.TrajectoryReader` is the lazy cursor over a store. The
``molrs.io.raw`` counterparts return eagerly.

A format whose native column names already are the canonical ones is not
wrapped: its reader and writer here *are* the compiled functions, and every
path argument takes a ``str`` or any ``os.PathLike``.

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
:meth:`Atomistic.def_port <molrs.Atomistic.def_port>` untranslated. The enums
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
lazy, path-backed trajectory cursor (:class:`TrajectoryReader`, and the
``*TrajReader`` classes of :mod:`molrs.io.raw`), and a text-in / IR-out parser
is not that object. A reader-shaped ``read()`` API over CGsmiles belongs to
molpy, which wraps :class:`CGSmilesIR` exactly as its ``SmilesReader`` wraps
:class:`SmilesIR`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any, Self, overload

from .._lib import (
    BondReactTemplate,
    BondingDescriptor,
    CGEdge,
    CGFragmentDef,
    CGGraph,
    CGNode,
    CGSmilesIR,
    Frame,
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
    mrec_sections,
    parse_lammps_log_text,
    read_ac,
    read_amber_inpcrd,
    read_amber_prmtop,
    read_chgcar,
    read_cube,
    read_frame,
    read_frame_bytes,
    read_lammps_data,
    read_lammps_log,
    read_mol2,
    read_mrec,
    read_mrec_forcefield,
    read_mrec_meta,
    read_mrec_system,
    read_mrec_trajectory,
    read_prep,
    read_stl,
    read_xsf,
    read_xyz,
    write_cube,
    write_dcd_trajectory,
    write_bond_react_map,
    write_frame,
    write_frame_bytes,
    write_lammps_bond_react_system,
    write_lammps_data,
    write_lammps_dump_local,
    write_lammps_molecule,
    write_lammps_trajectory,
    write_mol2,
    write_mrec,
    write_mrec_forcefield,
    write_mrec_system,
    write_mrec_trajectory,
    write_pdb,
    write_pdb_trajectory,
    write_prep,
    write_smarts,
    write_smiles,
    write_trr_trajectory,
    write_xsf,
    write_xtc_trajectory,
    write_xyz,
    write_xyz_trajectory,
)
from .._lib import DCDTrajReader as _DCDTrajReader
from .._lib import LAMMPSTrajReader as _LAMMPSTrajReader
from .._lib import TRRTrajReader as _TRRTrajReader
from .._lib import XTCTrajReader as _XTCTrajReader
from .._lib import XYZTrajReader as _XYZTrajReader
from .._lib import read_block_csv as _read_block_csv
from .._lib import read_gro as _read_gro
from .._lib import read_gro_trajectory as _read_gro_trajectory
from .._lib import read_lammps_molecule as _read_lammps_molecule
from .._lib import read_pdb as _read_pdb
from .._lib import read_pdb_trajectory as _read_pdb_trajectory
from .._lib import write_block_csv as _write_block_csv
from .._lib import write_gro as _write_gro
from .._lib import write_gro_trajectory as _write_gro_trajectory
from ..fields import (
    FieldFormatter,
    LammpsFieldFormatter,
    PdbFieldFormatter,
)
from . import mrec, raw

PathInput = str | PathLike[str]

_pdb_fmt = PdbFieldFormatter()
_lammps_fmt = LammpsFieldFormatter()
# XYZ, DCD, TRR and XTC readers already emit canonical names (DCD / TRR /
# XTC frames carry only coordinates and the box), so a no-op formatter is
# correct.
_noop_fmt = FieldFormatter()


def read_pdb(file: PathInput) -> Frame:
    """Read a PDB file.

    Returns a molrs ``Frame`` with canonical field names
    (``element`` instead of ``symbol``, ``res_id`` instead of ``resid``).
    """
    result = _read_pdb(file)
    _pdb_fmt.canonicalize_frame(result)
    return result


def read_pdb_trajectory(file: PathInput) -> list[Frame]:
    """Read every MODEL of a PDB file as a trajectory (one Frame per MODEL).

    A single-model (or MODEL-less) PDB returns a one-element list. Each frame
    is canonicalized like :func:`read_pdb`.
    """
    frames = _read_pdb_trajectory(file)
    for frame in frames:
        _pdb_fmt.canonicalize_frame(frame)
    return frames


def read_gro(file: PathInput) -> Frame:
    """Read the first frame of a GROMACS GRO file.

    Returns:
        A molrs ``Frame`` with canonical field names (``res_id``,
        ``res_name``, ``name``, ``element``, ``id``); coordinates in Å.

    Raises:
        OSError: If the file cannot be opened or parsed, or holds no frame.
    """
    return _read_gro(file)


def read_gro_trajectory(file: PathInput) -> list[Frame]:
    """Read every frame of a GROMACS GRO file, in file order.

    Each frame is canonicalized like :func:`read_gro`; a single-frame file
    returns a one-element list. Inverse of :func:`write_gro_trajectory`.
    """
    return _read_gro_trajectory(file)


def read_lammps_molecule(file: PathInput) -> Frame:
    """Read a LAMMPS molecule template (native or JSON).

    Returns a molrs ``Frame`` with canonical field names (``charge`` not ``q``).
    """
    result = _read_lammps_molecule(file)
    _lammps_fmt.canonicalize_frame(result)
    return result


def write_gro(file: PathInput, frame: Frame) -> None:
    """Write a single Frame to a GROMACS GRO file.

    Expects canonical columns (``res_id``, ``res_name``, ``name``, ``id``);
    coordinates in Å, written as nm. *frame* is left as it was passed.
    """
    _write_gro(file, frame)


def write_gro_trajectory(file: PathInput, frames: Sequence[Frame]) -> None:
    """Write Frames as one multi-frame GROMACS GRO trajectory.

    Each frame as :func:`write_gro` writes it; inverse of
    :func:`read_gro_trajectory`.
    """
    _write_gro_trajectory(file, list(frames))


# ===================================================================
#                     Trajectory readers
# ===================================================================


def _as_paths(file: PathInput | Sequence[PathInput]) -> list[PathInput]:
    """Normalise a single path or a sequence of paths to a list."""
    if isinstance(file, (str, PathLike)):
        return [file]
    return list(file)


class TrajectoryReader:
    """Lazy, indexed trajectory reader over one or more files.

    Wraps one or more native molrs readers (``DCDTrajReader``,
    ``LAMMPSTrajReader``, ``XYZTrajReader``, ``TRRTrajReader``,
    ``XTCTrajReader``). When constructed from several files their frames are
    concatenated into one logical trajectory. Every returned :class:`Frame` is
    canonicalized to project-wide field names.

    Surface: ``read_frame`` (negative indexing), ``read_frames``,
    ``read_range``, ``read_all``, ``n_frames``, integer and slice indexing,
    lazy iteration, ``close()``, and use as a context manager.
    """

    def __init__(self, readers: Sequence[Any], formatter: FieldFormatter) -> None:
        self._readers = list(readers)
        self._formatter = formatter
        self._counts: list[int] | None = None

    def _ensure_counts(self) -> list[int]:
        if self._counts is None:
            self._counts = [r.n_frames for r in self._readers]
        return self._counts

    def _locate(self, index: int) -> tuple[Any, int]:
        counts = self._ensure_counts()
        total = sum(counts)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError("trajectory index out of range")
        for reader, count in zip(self._readers, counts, strict=True):
            if index < count:
                return reader, index
            index -= count
        raise IndexError("trajectory index out of range")  # pragma: no cover

    @property
    def n_frames(self) -> int:
        return sum(self._ensure_counts())

    def read_frame(self, index: int) -> Frame:
        """Read a single frame (supports negative indexing).

        The single chokepoint for every indexed trajectory access
        (``read_frames`` / ``read_range`` / ``read_all`` / indexing all funnel
        here), so canonicalizing once here covers them all.
        """
        reader, local = self._locate(index)
        frame = reader.read_frame(local)
        self._formatter.canonicalize_frame(frame)
        return frame

    def read_frames(self, indices: Sequence[int]) -> list[Frame]:
        """Read an explicit list of frame indices."""
        return [self.read_frame(i) for i in indices]

    def read_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> list[Frame]:
        """Read a contiguous range of frames, Python-slice style."""
        if step == 0:
            raise ValueError("read_range step must not be zero")
        n = self.n_frames
        return [self.read_frame(i) for i in range(*slice(start, stop, step).indices(n))]

    def read_all(self) -> list[Frame]:
        """Eagerly read every frame into a list."""
        return [self.read_frame(i) for i in range(self.n_frames)]

    def close(self) -> None:
        """Release every underlying file handle."""
        for reader in self._readers:
            reader.close()

    def __len__(self) -> int:
        return self.n_frames

    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...

    def __getitem__(self, key: int | slice) -> Frame | list[Frame]:
        if isinstance(key, slice):
            return [self.read_frame(i) for i in range(*key.indices(self.n_frames))]
        return self.read_frame(key)

    def __iter__(self) -> Iterator[Frame]:
        """Sequential pass over all files, one frame at a time.

        Walks each native reader's own cursor, so it **does not** force a
        full-file index scan first; for TRR/XTC that is one contiguous read of
        the coordinate blocks. Ask ``n_frames`` yourself when a known length
        is needed (progress, preallocation).
        """
        for reader in self._readers:
            for frame in reader:
                self._formatter.canonicalize_frame(frame)
                yield frame

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"TrajectoryReader(n_frames={self.n_frames}, files={len(self._readers)})"


def read_lammps_trajectory(traj: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one LAMMPS dump file, or several whose frames are concatenated, as
    a lazy :class:`TrajectoryReader` with canonical field names."""
    return TrajectoryReader(
        [_LAMMPSTrajReader(p) for p in _as_paths(traj)], _lammps_fmt
    )


def read_xyz_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one XYZ trajectory, or several whose frames are concatenated, as a
    lazy :class:`TrajectoryReader` (``molrs.io.raw.read_xyz_trajectory`` is
    the eager ``list[Frame]`` form)."""
    return TrajectoryReader([_XYZTrajReader(p) for p in _as_paths(file)], _noop_fmt)


def read_dcd_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one DCD trajectory, or several whose frames are concatenated, as a
    lazy :class:`TrajectoryReader`."""
    return TrajectoryReader([_DCDTrajReader(p) for p in _as_paths(file)], _noop_fmt)


def read_trr_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one GROMACS TRR trajectory, or several whose frames are
    concatenated, as a lazy :class:`TrajectoryReader`. Random access is O(1)
    after a one-time index scan."""
    return TrajectoryReader([_TRRTrajReader(p) for p in _as_paths(file)], _noop_fmt)


def read_xtc_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one GROMACS XTC (compressed) trajectory, or several whose frames
    are concatenated, as a lazy :class:`TrajectoryReader`. Random access is
    O(1) after a one-time index scan."""
    return TrajectoryReader([_XTCTrajReader(p) for p in _as_paths(file)], _noop_fmt)


# ===================================================================
#                     Block CSV
# ===================================================================


def read_block_csv(
    source: PathInput | StringIO,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    header: list[str] | None = None,
    skip_empty_fields: bool = False,
) -> Any:
    """Read CSV into a :class:`Block`.

    Accepts the same sources every other reader in this module does — a path,
    or in-memory text via :class:`io.StringIO`. A bare ``str`` is a path when it
    names an existing file and CSV text otherwise.
    """
    if isinstance(source, StringIO):
        text = source.getvalue()
    else:
        path = Path(source)
        text = (
            path.read_text(encoding=encoding)
            if (isinstance(source, PathLike) or path.exists())
            else str(source)
        )
    d = delimiter if len(delimiter) == 1 else ","
    if skip_empty_fields:
        text = "\n".join(
            d.join(part for part in line.split(d) if part != "")
            for line in text.splitlines()
        )
    return _read_block_csv(text, d, header)


def write_block_csv(
    block: Any,
    filepath: PathInput | None = None,
    *,
    delimiter: str = ",",
    header: bool = True,
    encoding: str = "utf-8",
) -> str | None:
    """Write a :class:`Block` as CSV — the inverse of :func:`read_block_csv`.

    Returns the text when *filepath* is ``None``, else writes it and returns
    ``None``.
    """
    d = delimiter if len(delimiter) == 1 else ","
    text = _write_block_csv(block, d, header)
    if filepath is None:
        return text
    Path(filepath).write_text(text, encoding=encoding)
    return None


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
    "mrec_sections",
    "parse_lammps_log_text",
    "raw",
    "read_ac",
    "read_amber_inpcrd",
    "read_amber_prmtop",
    "read_block_csv",
    "read_chgcar",
    "read_cube",
    "read_dcd_trajectory",
    "read_frame",
    "read_frame_bytes",
    "read_gro",
    "read_gro_trajectory",
    "read_lammps_data",
    "read_lammps_log",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    "read_mol2",
    "read_mrec",
    "read_mrec_forcefield",
    "read_mrec_meta",
    "read_mrec_system",
    "read_mrec_trajectory",
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
    "write_frame_bytes",
    "write_gro",
    "write_gro_trajectory",
    "write_lammps_bond_react_system",
    "write_lammps_data",
    "write_lammps_dump_local",
    "write_lammps_molecule",
    "write_lammps_trajectory",
    "write_mol2",
    "write_mrec",
    "write_mrec_forcefield",
    "write_mrec_system",
    "write_mrec_trajectory",
    "write_pdb",
    "write_pdb_trajectory",
    "write_prep",
    "write_smarts",
    "write_smiles",
    "write_trr_trajectory",
    "write_xsf",
    "write_xtc_trajectory",
    "write_xyz",
    "write_xyz_trajectory",
]
