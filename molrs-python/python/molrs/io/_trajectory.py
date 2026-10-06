"""The lazy multi-file trajectory reader behind ``molrs.io.read_*_trajectory``.

Private: :mod:`molrs.io` is these names' public path.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Self, overload

from .._lib import DCDTrajReader as _DCDTrajReader
from .._lib import LAMMPSTrajReader as _LAMMPSTrajReader
from .._lib import TRRTrajReader as _TRRTrajReader
from .._lib import XTCTrajReader as _XTCTrajReader
from .._lib import XYZTrajReader as _XYZTrajReader

if TYPE_CHECKING:
    from ..store import Frame

PathInput = str | PathLike[str]


def _as_paths(file: PathInput | Sequence[PathInput]) -> list[PathInput]:
    """Normalise a single path or a sequence of paths to a list."""
    if isinstance(file, (str, PathLike)):
        return [file]
    return list(file)


class TrajectoryReader:
    """Lazy, indexed trajectory reader over one or more files.

    Wraps one or more native molrs readers (one per file, of the format the
    ``read_*_trajectory`` function that built it reads). When constructed from
    several files their frames are concatenated into one logical trajectory.

    Surface: ``read_frame`` (negative indexing), ``read_frames``,
    ``read_range``, ``read_all``, ``n_frames``, integer and slice indexing,
    lazy iteration, ``close()``, and use as a context manager.
    """

    def __init__(self, readers: Sequence[Any]) -> None:
        self._readers = list(readers)
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
        """Read a single frame (supports negative indexing)."""
        reader, local = self._locate(index)
        return reader.read_frame(local)

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
            yield from reader

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"TrajectoryReader(n_frames={self.n_frames}, files={len(self._readers)})"


def read_lammps_trajectory(traj: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one LAMMPS dump file, or several whose frames are concatenated, as
    a lazy :class:`TrajectoryReader`."""
    return TrajectoryReader([_LAMMPSTrajReader(p) for p in _as_paths(traj)])


def read_xyz_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one XYZ trajectory, or several whose frames are concatenated, as a
    lazy :class:`TrajectoryReader`."""
    return TrajectoryReader([_XYZTrajReader(p) for p in _as_paths(file)])


def read_dcd_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one DCD trajectory, or several whose frames are concatenated, as a
    lazy :class:`TrajectoryReader`."""
    return TrajectoryReader([_DCDTrajReader(p) for p in _as_paths(file)])


def read_trr_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one GROMACS TRR trajectory, or several whose frames are
    concatenated, as a lazy :class:`TrajectoryReader`. Random access is O(1)
    after a one-time index scan."""
    return TrajectoryReader([_TRRTrajReader(p) for p in _as_paths(file)])


def read_xtc_trajectory(file: PathInput | Sequence[PathInput]) -> TrajectoryReader:
    """Open one GROMACS XTC (compressed) trajectory, or several whose frames
    are concatenated, as a lazy :class:`TrajectoryReader`. Random access is
    O(1) after a one-time index scan."""
    return TrajectoryReader([_XTCTrajReader(p) for p in _as_paths(file)])
