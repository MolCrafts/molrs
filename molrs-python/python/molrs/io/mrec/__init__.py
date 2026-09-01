"""Scientific-record I/O for ``*.mrec`` stores.

``Frame`` and ``Trajectory`` are the in-memory objects. Path doors live here:

* :func:`read_frame` / :func:`write_frame` — Structure (``meta`` + ``frame/``)
* :func:`read_system` / :func:`write_system` — System-def (``meta`` + ``system/``)
* :func:`read_trajectory` / :func:`write_trajectory` — Trajectory shape
* :func:`read_meta` — identity document
* :func:`sections` — which groups are present at the root
* :class:`TrajectoryReader` — lazy frame cursor wrapping Rust
  ``FrameSequence`` (one frame per :meth:`TrajectoryReader.read_frame`; also
  ``len()``, ``reader[i]``, iteration, ``.step`` / ``.time`` labels and
  ``has_block``)
* :class:`SequenceSchema` / :class:`TrajectoryWriter` — pin a schema and write
  a run frame by frame, without holding it all in memory
* :mod:`molrs.io.mrec.schema` — runtime check for path suffix and ``meta`` brand

These names are not re-exported from :mod:`molrs.io`. That module's
:class:`~molrs.io.TrajectoryReader` remains the LAMMPS/XYZ/DCD dump
concatenator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import schema
from ..._lib import (
    MrecSequenceSchema as _MrecSequenceSchema,
    MrecTrajectoryReader as _MrecTrajectoryReader,
    MrecTrajectoryWriter as _MrecTrajectoryWriter,
    read_frame as _read_frame,
    read_meta as _read_meta,
    read_system as _read_system,
    read_trajectory as _read_trajectory,
    section_names as _section_names,
    write_frame as _write_frame,
    write_system as _write_system,
    write_trajectory as _write_trajectory,
)


def _path(path: str | Path) -> str:
    return str(Path(path).expanduser())


class TrajectoryReader:
    """Lazy one-frame cursor over a ``*.mrec`` trajectory.

    Wraps the Rust ``FrameSequence`` store cursor: construction opens the
    index only, and each read decodes exactly the asked-for frame. Supports
    ``len(reader)``, ``reader[i]`` (negative indices included), iteration, and
    the ``with`` statement.

    Args:
        path: Filesystem path of the record store.
    """

    def __init__(self, path: str | Path) -> None:
        self._inner = _MrecTrajectoryReader(_path(path))

    def read_frame(self, index: int):
        """Decode one committed frame.

        Args:
            index: Frame index; negative counts from the end.

        Returns:
            The frame at ``index``.
        """
        return self._inner.read_frame(index)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int):
        return self._inner[index]

    def __iter__(self):
        return iter(self._inner)

    @property
    def step(self) -> list[int]:
        """Per-frame step numbers — the frame labels a replay UI shows."""
        return self._inner.step

    @property
    def time(self) -> list[float] | None:
        """Per-frame physical times (fs), when the run wrote any."""
        return self._inner.time

    def has_block(self, name: str) -> bool:
        """Whether the store carries a block section (e.g. ``"bonds"``)."""
        return self._inner.has_block(name)

    def block_names(self) -> list[str]:
        """Names of every block section present in the store."""
        return self._inner.block_names()

    def __enter__(self) -> TrajectoryReader:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class SequenceSchema:
    """A schema pinned before a run's frames are written.

    Fixes every block, column and dtype a run may carry; a
    :class:`TrajectoryWriter` refuses a frame that steps outside it.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner

    @classmethod
    def from_frame(cls, frame) -> SequenceSchema:
        """Derive a schema from one representative frame."""
        return cls(_MrecSequenceSchema.from_frame(frame))

    @classmethod
    def from_frames(cls, frames) -> SequenceSchema:
        """Derive a schema from the union of several frames' blocks/columns.

        Pass a frame that carries an (even empty) ``bonds`` block to declare a
        section that only appears partway through a run.
        """
        return cls(_MrecSequenceSchema.from_frames(list(frames)))


class TrajectoryWriter:
    """Append-first writer for a ``*.mrec`` trajectory store.

    Wraps the Rust ``FrameSequenceWriter`` over the fast positional-write
    store, so a growing run is written frame by frame without holding the whole
    trajectory in memory. Use as a context manager, or call :meth:`close`.

    Args:
        path: Destination filesystem path for the store.
        schema: The :class:`SequenceSchema` every appended frame is checked
            against.
    """

    def __init__(self, path: str | Path, schema: SequenceSchema) -> None:
        self._inner = _MrecTrajectoryWriter(_path(path), schema._inner)

    def append(self, frame, step: int | None = None, time: float | None = None) -> None:
        """Buffer a frame. Omit ``step`` to number frames 0, 1, 2, …; pass it
        (with optional ``time`` in fs) for real MD numbering."""
        self._inner.append(frame, step, time)

    def flush(self) -> None:
        """Land every buffered frame on disk."""
        self._inner.flush()

    def close(self) -> None:
        """Flush and seal the store. Idempotent."""
        self._inner.close()

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def write_frame(
    path: str | Path,
    frame,
    system=None,
    meta: dict[str, Any] | None = None,
) -> None:
    _write_frame(_path(path), frame, system, meta)


def write_system(path: str | Path, system, meta: dict[str, Any] | None = None) -> None:
    _write_system(_path(path), system, meta)


def write_trajectory(path: str | Path, trajectory) -> None:
    _write_trajectory(_path(path), trajectory)


def read_frame(path: str | Path):
    return _read_frame(_path(path))


def read_system(path: str | Path):
    return _read_system(_path(path))


def read_trajectory(path: str | Path):
    return _read_trajectory(_path(path))


def read_meta(path: str | Path) -> dict[str, Any]:
    """Read the mandatory ``meta`` document of a ``*.mrec`` store."""
    return _read_meta(_path(path))


def sections(path: str | Path) -> frozenset[str]:
    """Child group names at the record root.

    Ask which sections are present; do not probe ``read_frame`` /
    ``read_system`` and catch a missing-section error.
    """
    return frozenset(_section_names(_path(path)))


__all__ = [
    "SequenceSchema",
    "TrajectoryReader",
    "TrajectoryWriter",
    "read_frame",
    "read_meta",
    "read_system",
    "read_trajectory",
    "schema",
    "sections",
    "write_frame",
    "write_system",
    "write_trajectory",
]
