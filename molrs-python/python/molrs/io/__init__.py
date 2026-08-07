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

Trajectories return a lazy :class:`TrajectoryReader` rather than a
``list[Frame]``: ``read_lammps_trajectory``, ``read_xyz_trajectory``,
``read_dcd_trajectory``, ``read_trr_trajectory``, ``read_xtc_trajectory``. Each
accepts a single path or a list of paths (frames are concatenated) and yields
canonical field names. The ``molrs.io.raw`` counterparts return eagerly.

:class:`SmilesIR` is here because SMILES is a *format*: text in, molecule out,
exactly like PDB or XYZ. SMARTS is not — a pattern is a query over a perceived
graph — so it lives in :mod:`molrs.perceive`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Any, Union, overload

from . import raw
from .._lib import SmilesIR as SmilesIR
from .._lib import (
    write_smiles as write_smiles,
    write_smarts as write_smarts,
)

from ..fields import (
    FieldFormatter,
    GroFieldFormatter,
    LammpsFieldFormatter,
    Mol2FieldFormatter,
    PdbFieldFormatter,
    XyzFieldFormatter,
)
from ..frame import Frame  # canonical rich Frame (spec frame-block-sink-01)
from .._lib import (
    DCDTrajReader as _DCDTrajReader,
    LAMMPSTrajReader as _LAMMPSTrajReader,
    TRRTrajReader as _TRRTrajReader,
    XTCTrajReader as _XTCTrajReader,
    XYZTrajReader as _XYZTrajReader,
    read_gro as _read_gro,
    read_lammps as _read_lammps,
    read_lammps_log as _read_lammps_log,
    parse_lammps_log_text as _parse_lammps_log_text,
    read_chgcar_file as _read_chgcar,
    read_cube_file as _read_cube,
    read_mol2 as _read_mol2,
    read_top as _read_top,
    write_top as _write_top,
    read_amber_inpcrd as _read_amber_inpcrd,
    read_amber_prmtop as _read_amber_prmtop,
    read_ac as _read_ac,
    read_frcmod as _read_frcmod,
    parse_frcmod as _parse_frcmod,
    write_frcmod as _write_frcmod,
    read_prep as _read_prep,
    write_prep as _write_prep,
    read_amber_prmtop_sections as _read_amber_prmtop_sections,
    prmtop_parse_pointers as _prmtop_parse_pointers,
    prmtop_parse_a4_names as _prmtop_parse_a4_names,
    prmtop_decode_bond_params as _prmtop_decode_bond_params,
    prmtop_decode_angle_params as _prmtop_decode_angle_params,
    prmtop_decode_dihedral_params as _prmtop_decode_dihedral_params,
    prmtop_decode_nonbond_params as _prmtop_decode_nonbond_params,
    read_lammps_molecule as _read_lammps_molecule,
    read_pdb as _read_pdb,
    read_pdb_trajectory as _read_pdb_trajectory,
    read_trr as _read_trr,
    read_xtc as _read_xtc,
    read_xyz as _read_xyz,
    read_xsf as _read_xsf,
    write_gro as _write_gro,
    write_lammps as _write_lammps,
    write_cube_file as _write_cube,
    write_mol2 as _write_mol2,
    write_lammps_molecule as _write_lammps_molecule,
    write_pdb as _write_pdb,
    write_pdb_trajectory as _write_pdb_trajectory,
    write_trr as _write_trr,
    write_xtc as _write_xtc,
    write_xyz as _write_xyz,
    write_xsf as _write_xsf,
)

_gro_fmt = GroFieldFormatter()
_pdb_fmt = PdbFieldFormatter()
_lammps_fmt = LammpsFieldFormatter()
_xyz_fmt = XyzFieldFormatter()
_mol2_fmt = Mol2FieldFormatter()
# DCD frames carry only coordinates / box — no format-specific column names to
# canonicalize, so a no-op formatter is correct.
_noop_fmt = FieldFormatter()

PathInput = Union[str, "PathLike[str]"]


def _wrap(frame: Any) -> Frame:
    """Upgrade a freshly-read, canonicalized bare frame to the rich :class:`Frame`.

    Zero-copy: the rich Frame views the same Rust-backed Block buffers (no
    column data is copied). Already-rich frames pass through unchanged.
    """
    return Frame.from_dict(frame)


def read_lammps_data(
    file: str | PathLike[str],
    atom_style: str | None = None,
    frame: Any = None,
) -> Any:
    """Read a LAMMPS data file. molpy-compatible signature.

    Args:
        file: Path to the LAMMPS data file.
        atom_style: Accepted for API parity with molpy; the Rust reader
            auto-detects the style from the file (column count and the
            optional ``Atoms # <style>`` comment).
        frame: Reserved for API parity. molrs always returns a new
            ``Frame``; passing an existing frame is not supported.

    Returns:
        A molrs ``Frame`` with canonical field names.
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_lammps_data does not accept an existing frame; "
            "it always returns a new Frame."
        )
    result = _read_lammps(str(file))
    _lammps_fmt.canonicalize_frame(result)
    return _wrap(result)


def read_pdb(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read a PDB file. molpy-compatible signature.

    Returns a molrs ``Frame`` with canonical field names
    (``element`` instead of ``symbol``, ``res_id`` instead of ``resid``).
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_pdb does not accept an existing frame."
        )
    result = _read_pdb(str(file))
    _pdb_fmt.canonicalize_frame(result)
    return _wrap(result)


def read_pdb_trajectory(file: str | PathLike[str]) -> list[Any]:
    """Read every MODEL of a PDB file as a trajectory (one Frame per MODEL).

    A single-model (or MODEL-less) PDB returns a one-element list. Each frame
    is canonicalized like :func:`read_pdb`.
    """
    frames = _read_pdb_trajectory(str(file))
    for frame in frames:
        _pdb_fmt.canonicalize_frame(frame)
    return [_wrap(frame) for frame in frames]


def read_xyz(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read an XYZ file. molpy-compatible signature.

    Returns a molrs ``Frame`` with canonical field names
    (``element`` instead of ``symbol``).
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_xyz does not accept an existing frame."
        )
    result = _read_xyz(str(file))
    _xyz_fmt.canonicalize_frame(result)
    return _wrap(result)


def read_gro(file: str | PathLike[str]) -> list[Any]:
    """Read all frames from a GROMACS GRO file.

    Args:
        file: Path to a ``.gro`` file (single- or multi-frame).

    Returns:
        List of molrs ``Frame`` objects with canonical field names
        (``res_id``, ``res_name``, ``name``, ``id``).

    Raises:
        OSError: If the file cannot be opened or parsed.
    """
    frames = _read_gro(str(file))
    for f in frames:
        _gro_fmt.canonicalize_frame(f)
    return [_wrap(f) for f in frames]


def read_chgcar(file: str | PathLike[str]) -> Any:
    """Read a VASP CHGCAR into a frame carrying a ``"chgcar"`` grid block.

    Args:
        file: Path to a ``CHGCAR`` file.

    Returns:
        A molrs ``Frame`` whose ``"chgcar"`` block holds the flattened grid
        (at least ``"total"``; spin-polarised files add ``"diff"``).

    Raises:
        OSError: If the file cannot be opened or parsed.
    """
    return _wrap(_read_chgcar(str(file)))


def read_cube(file: str | PathLike[str]) -> Any:
    """Read a Gaussian Cube file into a frame carrying a grid block.

    Args:
        file: Path to a ``.cube`` file.

    Returns:
        A molrs ``Frame`` with the volumetric data as a flattened grid block.

    Raises:
        OSError: If the file cannot be opened or parsed.
    """
    return _wrap(_read_cube(str(file)))


def write_cube(file: str | PathLike[str], frame: Any) -> None:
    """Write a frame's grid block to a Gaussian Cube file.

    Args:
        file: Destination ``.cube`` path.
        frame: A molrs ``Frame`` carrying a grid block.

    Raises:
        OSError: If the file cannot be written.
    """
    _write_cube(str(file), frame)


def read_mol2(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read a Tripos MOL2 file (first molecule).

    Returns a molrs ``Frame`` with canonical field names (``type`` instead of
    ``atom_type``, ``res_id``/``res_name`` instead of ``subst_*``).
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_mol2 does not accept an existing frame; "
            "it always returns a new Frame."
        )
    result = _read_mol2(str(file))
    _mol2_fmt.canonicalize_frame(result)
    return _wrap(result)


def read_amber_inpcrd(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read an AMBER ASCII inpcrd / restrt coordinate file.

    Returns a molrs ``Frame`` with ``id``, ``name``, ``x``/``y``/``z``,
    optional ``vel`` (shape ``[n, 3]``), optional box, and meta ``title`` /
    ``timestep``. Always returns a new Frame; *frame* is accepted for API
    parity and must be ``None``.

    Args:
        file: Path to a ``.inpcrd`` or restart file.
        frame: Reserved for API parity. Passing an existing frame raises.

    Returns:
        A molrs ``Frame`` with coordinates (Å).

    Raises:
        NotImplementedError: If *frame* is not ``None``.
        OSError: If the file cannot be opened or parsed.
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_amber_inpcrd does not accept an existing frame; "
            "it always returns a new Frame."
        )
    return _wrap(_read_amber_inpcrd(str(file)))


def read_inpcrd(file: str | PathLike[str], frame: Any = None) -> Any:
    """Alias for :func:`read_amber_inpcrd`."""
    return read_amber_inpcrd(file, frame=frame)


def read_amber_prmtop(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read an AMBER prmtop **structure** file into a Frame.

    Structure / connectivity only: atoms (``name``, ``type``, ``charge`` in
    electron units, ``mass``, optional ``atomic_number``/``element``,
    ``res_id``), bonds/angles/dihedrals (0-based indices, type labels), and
    POINTERS meta. Force-field parameter tables are not assembled.

    Always returns a new Frame; *frame* is accepted for API parity and must be
    ``None``.

    Args:
        file: Path to a ``.prmtop`` / ``.parm7`` file.
        frame: Reserved for API parity. Passing an existing frame raises.

    Returns:
        A molrs ``Frame`` with topology blocks and POINTERS meta.

    Raises:
        NotImplementedError: If *frame* is not ``None``.
        OSError: If the file cannot be opened or parsed.
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_amber_prmtop does not accept an existing frame; "
            "it always returns a new Frame."
        )
    return _wrap(_read_amber_prmtop(str(file)))


def read_prmtop(file: str | PathLike[str], frame: Any = None) -> Any:
    """Alias for :func:`read_amber_prmtop`."""
    return read_amber_prmtop(file, frame=frame)


def write_mol2(file: str | PathLike[str], frame: Any) -> None:
    """Write a Frame to a Tripos MOL2 file.

    Localises canonical columns (``type`` → ``atom_type``, residue pair) before
    the native writer.
    """
    # Localise on a shallow view: rename is in-place on blocks; callers that
    # need the canonical frame afterwards should copy first.
    _mol2_fmt.localize_frame(frame)
    try:
        _write_mol2(str(file), frame)
    finally:
        # Restore canonical names so a write does not permanently mutate.
        _mol2_fmt.canonicalize_frame(frame)


def read_top(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read a GROMACS topology (``.top``) structure file.

    Structure only: ``[ atoms ]``, ``[ bonds ]``, ``[ pairs ]``,
    ``[ angles ]``, ``[ dihedrals ]``. Connectivity indices are 1-based as in
    the file. ``#include`` is not expanded.

    Returns a molrs ``Frame`` (no field renames — top-native names already match
    the historical molpy structure contract).
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_top does not accept an existing frame; "
            "it always returns a new Frame."
        )
    return _wrap(_read_top(str(file)))


def write_top(file: str | PathLike[str], frame: Any) -> None:
    """Write a Frame as a minimal GROMACS topology structure file.

    Molecule name from ``frame.meta["name"]`` (default ``"MOL"``). Connectivity
    atom indices written as stored (1-based contract).
    """
    _write_top(str(file), frame)


def read_lammps_molecule(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read a LAMMPS molecule template (native or JSON).

    Returns a molrs ``Frame`` with canonical field names (``charge`` not ``q``).
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_lammps_molecule does not accept an existing frame; "
            "it always returns a new Frame."
        )
    result = _read_lammps_molecule(str(file))
    _lammps_fmt.canonicalize_frame(result)
    return _wrap(result)


def write_lammps_molecule(
    file: str | PathLike[str],
    frame: Any,
    format: str = "native",
) -> None:
    """Write a Frame as a LAMMPS molecule template.

    Args:
        file: Output path.
        frame: Molecule frame (canonical columns).
        format: ``"native"`` or ``"json"``.
    """
    _write_lammps_molecule(str(file), frame, format)


def read_trr(file: str | PathLike[str]) -> list[Any]:
    """Read all frames from a GROMACS TRR trajectory.

    Args:
        file: Path to a ``.trr`` file.

    Returns:
        List of molrs ``Frame`` objects (``id``, ``x``/``y``/``z`` in nm, plus
        ``vx``/``vy``/``vz`` and ``fx``/``fy``/``fz`` when present).

    Raises:
        OSError: If the file cannot be opened or parsed.
    """
    frames = _read_trr(str(file))
    for f in frames:
        _noop_fmt.canonicalize_frame(f)
    return [_wrap(f) for f in frames]


def read_xtc(file: str | PathLike[str]) -> list[Any]:
    """Read all frames from a GROMACS XTC (compressed) trajectory.

    Args:
        file: Path to a ``.xtc`` file (classic 1995 or 2023 magic).

    Returns:
        List of molrs ``Frame`` objects (``id``, ``x``/``y``/``z`` in nm).

    Raises:
        OSError: If the file cannot be opened or parsed.
    """
    frames = _read_xtc(str(file))
    for f in frames:
        _noop_fmt.canonicalize_frame(f)
    return [_wrap(f) for f in frames]


def write_lammps_data(
    file: str | PathLike[str],
    frame: Any,
    atom_style: str | None = None,
) -> None:
    """Write a LAMMPS data file. molpy-compatible signature.

    ``atom_style`` is accepted for API parity but the writer derives the
    style from the columns present in ``frame['atoms']``.

    The frame must use **canonical** column names (``type_id``, ``mol_id``,
    ``charge``, …). The Rust writer reads those keys directly — do **not**
    localise to LAMMPS file spellings (``type``/``mol``/``q``); that renamed
    ``type_id`` onto the string-typed schema key ``type`` and broke writes.
    """
    del atom_style  # API parity only
    _write_lammps(str(file), frame)


def write_pdb(file: str | PathLike[str], frame: Any) -> None:
    """Write a PDB file.

    Expects **canonical** columns (``element``, ``res_name``, ``res_id``, …).
    Localising to PDB-native names (``symbol``/``resname``) was a bug: the
    Rust writer already uses the canonical vocabulary.
    """
    _write_pdb(str(file), frame)


def write_pdb_trajectory(file: str | PathLike[str], frames: Any) -> None:
    """Write a list of Frames as a multi-MODEL PDB trajectory.

    Each frame becomes one ``MODEL``/``ENDMDL`` block. Frames must use
    canonical column names (see :func:`write_pdb`).
    """
    _write_pdb_trajectory(str(file), list(frames))


def write_xyz(file: str | PathLike[str], frame: Any) -> None:
    """Write an XYZ file.

    Expects the canonical ``element`` column. Localising to ``symbol`` broke
    the Rust writer, which looks up ``element``.
    """
    _write_xyz(str(file), frame)


def read_xsf(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read an XSF (XCrySDen) structure file.

    Returns a molrs ``Frame`` with ``atomic_number``, ``element``, and
    ``x``/``y``/``z``. Crystal structures carry a periodic box; molecules a
    free box.
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_xsf does not accept an existing frame; "
            "it always returns a new Frame."
        )
    return _wrap(_read_xsf(str(file)))


def write_xsf(file: str | PathLike[str], frame: Any) -> None:
    """Write a Frame to an XSF (XCrySDen) structure file.

    Expects ``atomic_number`` and ``x``/``y``/``z`` on the atoms block. A
    defined non-free box is written as ``CRYSTAL``; otherwise ``MOLECULE``.
    """
    _write_xsf(str(file), frame)


def write_gro(file: str | PathLike[str], frame: Any) -> None:
    """Write a single Frame to a GROMACS GRO file.

    Localises *frame* in-place (``res_id`` → ``resid``,
    ``name`` → ``atom_name``, ``id`` → ``atom_id``) before writing.
    """
    _gro_fmt.localize_frame(frame)
    _write_gro(str(file), frame)


def write_trr(file: str | PathLike[str], frames: Any) -> None:
    """Write a list of Frames to a GROMACS TRR trajectory (single precision).

    Each frame needs ``x``/``y``/``z`` (nm); optional ``vx``/``vy``/``vz`` and
    ``fx``/``fy``/``fz`` are written when present.
    """
    _write_trr(str(file), list(frames))


def write_xtc(file: str | PathLike[str], frames: Any) -> None:
    """Write a list of Frames to a GROMACS XTC trajectory (lossy compression).

    Each frame needs ``x``/``y``/``z`` (nm); the quantization precision is taken
    from ``frame.meta['precision']`` if present, else 1000 (0.001 nm).
    """
    _write_xtc(str(file), list(frames))


# ===================================================================
#                     Trajectory readers
# ===================================================================


def _as_paths(file: PathInput | Sequence[PathInput]) -> list[str]:
    """Normalise a single path or a sequence of paths to ``list[str]``."""
    if isinstance(file, (str, PathLike)):
        return [str(file)]
    return [str(p) for p in file]


class TrajectoryReader:
    """molpy-compatible lazy, indexed trajectory reader.

    Wraps one or more native molrs readers (``DCDTrajReader``,
    ``LAMMPSTrajReader``, ``XYZTrajReader``). When constructed from several
    files their frames are concatenated into one logical trajectory. Every
    returned :class:`Frame` is canonicalized to project-wide field names.

    Mirrors molpy's ``BaseTrajectoryReader``: ``read_frame`` (negative
    indexing), ``read_frames``, ``read_range``, ``read_all``, ``n_frames``,
    integer and slice indexing, lazy iteration, ``close()``, and use as a
    context manager.
    """

    def __init__(self, readers: Sequence[Any], formatter: FieldFormatter) -> None:
        self._readers = list(readers)
        self._formatter = formatter
        self._counts: list[int] | None = None
        self._cursor = 0

    # ── internal ──────────────────────────────────────────────────

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
        for reader, count in zip(self._readers, counts):
            if index < count:
                return reader, index
            index -= count
        raise IndexError("trajectory index out of range")  # pragma: no cover

    # ── molpy BaseTrajectoryReader surface ────────────────────────

    @property
    def n_frames(self) -> int:
        return sum(self._ensure_counts())

    def read_frame(self, index: int) -> Frame:
        """Read a single frame (supports negative indexing).

        The single chokepoint for every trajectory access (``read_frames`` /
        ``read_range`` / ``read_all`` / indexing / iteration all funnel here),
        so wrapping to the rich :class:`Frame` once here covers them all.
        """
        reader, local = self._locate(index)
        frame = reader.read_frame(local)
        self._formatter.canonicalize_frame(frame)
        return _wrap(frame)

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
            n = self.n_frames
            return [self.read_frame(i) for i in range(*key.indices(n))]
        return self.read_frame(key)

    def __iter__(self) -> Iterator[Frame]:
        self._cursor = 0
        return self

    def __next__(self) -> Frame:
        if self._cursor >= self.n_frames:
            raise StopIteration
        frame = self.read_frame(self._cursor)
        self._cursor += 1
        return frame

    def __enter__(self) -> "TrajectoryReader":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"TrajectoryReader(n_frames={self.n_frames}, files={len(self._readers)})"


def read_lammps_trajectory(
    traj: PathInput | Sequence[PathInput], frame: Any = None
) -> TrajectoryReader:
    """Open a LAMMPS dump trajectory. molpy-compatible signature.

    Args:
        traj: Path or list of paths to LAMMPS dump files.
        frame: Reserved for molpy API parity; not supported.

    Returns:
        A lazy :class:`TrajectoryReader` with canonical field names.
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_lammps_trajectory does not accept a reference frame."
        )
    readers = [_LAMMPSTrajReader(p) for p in _as_paths(traj)]
    return TrajectoryReader(readers, _lammps_fmt)


def read_xyz_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open an XYZ trajectory. molpy-compatible signature.

    Unlike ``molrs.io.raw.read_xyz_trajectory`` (which returns
    ``list[Frame]``), this returns a lazy :class:`TrajectoryReader`, matching
    molpy's ``read_xyz_trajectory``.
    """
    readers = [_XYZTrajReader(p) for p in _as_paths(file)]
    return TrajectoryReader(readers, _xyz_fmt)


def read_dcd_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open a DCD trajectory as a lazy :class:`TrajectoryReader`.

    molrs extension (molpy has no DCD reader). Accepts a single path or a
    list of paths whose frames are concatenated.
    """
    readers = [_DCDTrajReader(p) for p in _as_paths(file)]
    return TrajectoryReader(readers, _noop_fmt)


def read_trr_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open a GROMACS TRR trajectory as a lazy :class:`TrajectoryReader`.

    Accepts a single path or a list of paths whose frames are concatenated.
    Random access is O(1) after a one-time index scan.
    """
    readers = [_TRRTrajReader(p) for p in _as_paths(file)]
    return TrajectoryReader(readers, _noop_fmt)


def read_xtc_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open a GROMACS XTC (compressed) trajectory as a lazy :class:`TrajectoryReader`.

    Accepts a single path or a list of paths whose frames are concatenated.
    Random access is O(1) after a one-time index scan.
    """
    readers = [_XTCTrajReader(p) for p in _as_paths(file)]
    return TrajectoryReader(readers, _noop_fmt)


def read_lammps_log(
    file: PathInput,
    style: str = "default",
) -> dict[str, Any]:
    """Read a LAMMPS log file into a nested plain dict.

    Parses thermo tables, loop timing, performance, CPU/MPI timing,
    load-balance stats, neighbor statistics, and warnings. Unrecognized
    lines are retained under each run's ``unparsed_log``.

    Args:
        file: Path to a LAMMPS log (e.g. ``log.lammps``).
        style: Thermo style. Only ``"default"`` is currently parsed.

    Returns:
        Nested mapping suitable for JSON / dataclass hydration. Thermo
        rows are ``list[list[float]]``.

    Raises:
        FileNotFoundError: If ``file`` does not exist.
    """
    return _read_lammps_log(str(file), style)


def parse_lammps_log_text(
    text: str,
    path: str = "<string>",
    style: str = "default",
) -> dict[str, Any]:
    """Parse a LAMMPS log from an in-memory string (no filesystem access).

    Args:
        text: Full log contents.
        path: Value recorded on the result as ``path``.
        style: Thermo style. Only ``"default"`` is currently parsed.

    Returns:
        Same nested shape as :func:`read_lammps_log`.
    """
    return _parse_lammps_log_text(text, path, style)


def read_ac(path: PathInput) -> Frame:
    """Read an Antechamber ``.ac`` file into a canonical Frame."""
    return _wrap(_read_ac(str(path)))


def read_frcmod(path: PathInput) -> dict[str, str]:
    """Read an AMBER FRCMOD file into a section dictionary."""
    return _read_frcmod(str(path))


def parse_frcmod(text: str) -> dict[str, str]:
    """Parse FRCMOD text into a section dictionary."""
    return _parse_frcmod(text)


def write_frcmod(path: PathInput, sections: dict[str, str]) -> None:
    """Write FRCMOD sections to *path*."""
    _write_frcmod(str(path), sections)


def read_prep(path: PathInput) -> dict:
    """Read an Amber prep file into a nested dict."""
    return _read_prep(str(path))


def write_prep(path: PathInput, residue: dict) -> None:
    """Write an Amber prep residue dict to *path*."""
    _write_prep(str(path), residue)


def read_amber_prmtop_sections(path: PathInput) -> dict[str, list[str]]:
    """Read raw prmtop ``%FLAG`` sections as ``{flag: [lines...]}``."""
    return _read_amber_prmtop_sections(str(path))


def prmtop_parse_pointers(lines: list[str]) -> dict[str, int]:
    """Parse POINTERS lines into the historical meta map."""
    return _prmtop_parse_pointers(list(lines))


def prmtop_parse_a4_names(lines: list[str]) -> list[str]:
    """Parse Fortran ``20a4`` name fields from section lines."""
    return _prmtop_parse_a4_names(list(lines))


def prmtop_decode_bond_params(
    pointers: list[int], force_k: list[float], equil: list[float]
) -> list[tuple]:
    """Decode bond tables → ``(type, i, j, K, r0)`` (atoms 1-based)."""
    return _prmtop_decode_bond_params(list(pointers), list(force_k), list(equil))


def prmtop_decode_angle_params(
    pointers: list[int], force_k: list[float], equil_rad: list[float]
) -> list[tuple]:
    """Decode angle tables → ``(type, i, j, k, K, theta0_deg)`` (1-based)."""
    return _prmtop_decode_angle_params(list(pointers), list(force_k), list(equil_rad))


def prmtop_decode_dihedral_params(
    pointers: list[int],
    force_k: list[float],
    phase: list[float],
    periodicity: list[float],
) -> list[tuple]:
    """Decode dihedral tables → ``(type, i, j, k, l, K, phase, n)`` (1-based)."""
    return _prmtop_decode_dihedral_params(
        list(pointers), list(force_k), list(phase), list(periodicity)
    )


def prmtop_decode_nonbond_params(
    n_atom: int,
    n_types: int,
    atom_type_index: list[int],
    nonbonded_parm_index: list[int],
    acoef: list[float],
    bcoef: list[float],
    hbond_a: list[float] | None = None,
    hbond_b: list[float] | None = None,
) -> list[tuple]:
    """Per-atom LJ ``(atom_1based, sigma, epsilon)``."""
    return _prmtop_decode_nonbond_params(
        n_atom,
        n_types,
        list(atom_type_index),
        list(nonbonded_parm_index),
        list(acoef),
        list(bcoef),
        list(hbond_a or []),
        list(hbond_b or []),
    )


__all__ = [
    "raw",
    "SmilesIR",
    "read_lammps_data",
    "read_pdb",
    "read_pdb_trajectory",
    "read_xyz",
    "read_gro",
    "read_chgcar",
    "read_cube",
    "read_mol2",
    "read_top",
    "read_amber_inpcrd",
    "read_inpcrd",
    "read_amber_prmtop",
    "read_ac",
    "read_frcmod",
    "parse_frcmod",
    "write_frcmod",
    "read_prep",
    "write_prep",
    "read_amber_prmtop_sections",
    "prmtop_parse_pointers",
    "prmtop_parse_a4_names",
    "prmtop_decode_bond_params",
    "prmtop_decode_angle_params",
    "prmtop_decode_dihedral_params",
    "prmtop_decode_nonbond_params",
    "read_prmtop",
    "read_lammps_molecule",
    "read_lammps_log",
    "parse_lammps_log_text",
    "read_xsf",
    "read_trr",
    "read_xtc",
    "write_lammps_data",
    "write_pdb",
    "write_pdb_trajectory",
    "write_xyz",
    "write_gro",
    "write_cube",
    "write_mol2",
    "write_top",
    "write_lammps_molecule",
    "write_xsf",
    "write_trr",
    "write_xtc",
    "write_smiles",
    "write_smarts",
    "TrajectoryReader",
    "read_lammps_trajectory",
    "read_xyz_trajectory",
    "read_dcd_trajectory",
    "read_trr_trajectory",
    "read_xtc_trajectory",
]
