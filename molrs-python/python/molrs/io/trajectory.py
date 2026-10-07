"""Trajectory files read lazily — :class:`TrajectoryReader`.

``molrs.io.read_lammps_trajectory``, ``read_xyz_trajectory``,
``read_dcd_trajectory``, ``read_trr_trajectory`` and ``read_xtc_trajectory``
open one (or several, concatenated) files as a :class:`TrajectoryReader`; the
functions are :mod:`molrs.io`'s, the reader class is this module's.
"""

from collections.abc import Iterator as _Iterator
from collections.abc import Sequence as _Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any
from typing import Self as _Self
from typing import overload as _overload

if _TYPE_CHECKING:
    from ..store import Frame

__all__ = ["TrajectoryReader"]


class TrajectoryReader:
    """Lazy, indexed trajectory reader over one or more files.

    Wraps one or more native molrs readers (one per file, of the format the
    ``read_*_trajectory`` function that built it reads). When constructed from
    several files their frames are concatenated into one logical trajectory.

    Surface: ``read_frame`` (negative indexing), ``read_frames``,
    ``read_range``, ``read_all``, ``n_frames``, integer and slice indexing,
    lazy iteration, ``close()``, and use as a context manager.
    """

    def __init__(self, readers: _Sequence[_Any]) -> None:
        self._readers = list(readers)
        self._counts: list[int] | None = None

    def _ensure_counts(self) -> list[int]:
        if self._counts is None:
            self._counts = [r.n_frames for r in self._readers]
        return self._counts

    def _locate(self, index: int) -> tuple[_Any, int]:
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

    def read_frame(self, index: int) -> "Frame":
        """Read a single frame (supports negative indexing)."""
        reader, local = self._locate(index)
        return reader.read_frame(local)

    def read_frames(self, indices: _Sequence[int]) -> "list[Frame]":
        """Read an explicit list of frame indices."""
        return [self.read_frame(i) for i in indices]

    def read_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> "list[Frame]":
        """Read a contiguous range of frames, Python-slice style."""
        if step == 0:
            raise ValueError("read_range step must not be zero")
        n = self.n_frames
        return [self.read_frame(i) for i in range(*slice(start, stop, step).indices(n))]

    def read_all(self) -> "list[Frame]":
        """Eagerly read every frame into a list."""
        return [self.read_frame(i) for i in range(self.n_frames)]

    def close(self) -> None:
        """Release every underlying file handle."""
        for reader in self._readers:
            reader.close()

    def __len__(self) -> int:
        return self.n_frames

    @_overload
    def __getitem__(self, key: int) -> "Frame": ...
    @_overload
    def __getitem__(self, key: slice) -> "list[Frame]": ...

    def __getitem__(self, key: int | slice) -> "Frame | list[Frame]":
        if isinstance(key, slice):
            return [self.read_frame(i) for i in range(*key.indices(self.n_frames))]
        return self.read_frame(key)

    def __iter__(self) -> "_Iterator[Frame]":
        """Sequential pass over all files, one frame at a time.

        Walks each native reader's own cursor, so it **does not** force a
        full-file index scan first; for TRR/XTC that is one contiguous read of
        the coordinate blocks. Ask ``n_frames`` yourself when a known length
        is needed (progress, preallocation).
        """
        for reader in self._readers:
            yield from reader

    def __enter__(self) -> _Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"TrajectoryReader(n_frames={self.n_frames}, files={len(self._readers)})"
