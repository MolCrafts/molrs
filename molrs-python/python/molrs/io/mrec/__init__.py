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
* :func:`pack` — collapse a closed store into one ``*.mrec.zip``
* :mod:`molrs.io.mrec.schema` — runtime check for path suffix and ``meta`` keys

These names are not re-exported from :mod:`molrs.io`. That module's
:class:`~molrs.io.TrajectoryReader` remains the LAMMPS/XYZ/DCD dump
concatenator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from ..._lib import (
    MrecSequenceSchema as _MrecSequenceSchema,
)
from ..._lib import (
    MrecTrajectoryReader as _MrecTrajectoryReader,
)
from ..._lib import (
    MrecTrajectoryWriter as _MrecTrajectoryWriter,
)
from ..._lib import (
    pack as _pack,
)
from ..._lib import (
    read_frame as _read_frame,
)
from ..._lib import (
    read_meta as _read_meta,
)
from ..._lib import (
    read_system as _read_system,
)
from ..._lib import (
    read_trajectory as _read_trajectory,
)
from ..._lib import (
    section_names as _section_names,
)
from ..._lib import (
    write_frame as _write_frame,
)
from ..._lib import (
    write_system as _write_system,
)
from ..._lib import (
    write_trajectory as _write_trajectory,
)
from . import schema


def _path(path: str | Path) -> str:
    return str(Path(path).expanduser())


class TrajectoryReader:
    """Lazy one-frame cursor over a ``*.mrec`` trajectory (directory or ``.zip``).

    Wraps the Rust ``FrameSequence`` store cursor: construction opens the
    index only, and each read decodes exactly the asked-for frame, keeping the
    last decoded chunk of every column so playback through consecutive frames
    is a slice rather than a decode. Supports ``len(reader)``, ``reader[i]``
    (negative indices included), iteration, and the ``with`` statement.

    Args:
        path: Filesystem path of the record store, or of a packed
            ``*.mrec.zip``.
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

    def read_columns(self, index: int, columns: list[tuple[str, str]]):
        """Decode one frame carrying only the named ``(block, column)`` pairs.

        A viewer that needs coordinates asks for ``[("atoms", "x"),
        ("atoms", "y"), ("atoms", "z")]`` and pays for nothing else. The cell
        and per-step metadata always come along.
        """
        return self._inner.read_columns(index, list(columns))

    def block_update_at(self, name: str, index: int) -> int | None:
        """The update of block ``name`` that frame ``index`` resolves to.

        ``None`` when the block is absent there. Two frames resolving to the
        same update carry the same rows, so a consumer can skip re-uploading
        a block that did not change without comparing values.
        """
        return self._inner.block_update_at(name, index)

    def box_at(self, index: int):
        """The cell at frame ``index``, or ``None`` before any cell was written."""
        return self._inner.box_at(index)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int):
        return self._inner[index]

    def __iter__(self):
        for index in range(len(self._inner)):
            yield self._inner.read_frame(index)

    @property
    def step(self) -> list[int]:
        """Per-frame step numbers — the frame labels a replay UI shows."""
        return self._inner.step

    @property
    def time(self) -> list[float] | None:
        """Per-frame physical times (fs), when the run wrote any."""
        return self._inner.time

    def has_block(self, name: str) -> bool:
        """Whether the store carries a block section of this name."""
        return self._inner.has_block(name)

    def block_names(self) -> list[str]:
        """Names of every block section present in the store."""
        return self._inner.block_names()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class SequenceSchema:
    """A schema pinned before a run's frames are written.

    Fixes every block, column and dtype a run may carry; a
    :class:`TrajectoryWriter` refuses a frame that steps outside it. Declare
    it column by column (``SequenceSchema()`` then ``declare_column`` …) or
    derive it from representative frames.
    """

    def __init__(self, inner: object | None = None) -> None:
        self._inner = inner if inner is not None else _MrecSequenceSchema()

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

    def declare_block(self, name: str, rows: int | None = None) -> SequenceSchema:
        """Declare a block, with the row count a typical frame of it carries."""
        self._inner.declare_block(name, rows)
        return self

    def declare_column(
        self,
        block: str,
        column: str,
        dtype: str,
        trailing: list[int] | None = None,
    ) -> SequenceSchema:
        """Declare a column by dtype tag (``"f64"``, ``"u64"``, ``"string"``, …)
        and trailing shape (``[3]`` for an xyz column)."""
        self._inner.declare_column(block, column, dtype, trailing)
        return self

    def declare_structural_shape(self, block: str, shape: list[int]) -> SequenceSchema:
        """Declare a volumetric block's shape; every update carries ``prod(shape)`` rows."""
        self._inner.declare_structural_shape(block, list(shape))
        return self

    def declare_meta(self, key: str, dtype: str) -> SequenceSchema:
        """Declare a per-step metadata key by dtype tag (``"f64"``, ``"f64x3"``, ``"json"`` …)."""
        self._inner.declare_meta(key, dtype)
        return self

    def declare_meta_with_fill(
        self, key: str, fill: Any, dtype: str | None = None
    ) -> SequenceSchema:
        """Declare a per-step metadata key with the value written when a frame
        omits it; ``dtype`` (``"f64x3"``, ``"f32"``, …) fixes the width, else
        the fill's own is used."""
        self._inner.declare_meta_with_fill(key, fill, dtype)
        return self

    def block_names(self) -> list[str]:
        return self._inner.block_names()

    def column_names(self, block: str) -> list[str] | None:
        return self._inner.column_names(block)

    def meta_keys(self) -> list[tuple[str, str]]:
        return self._inner.meta_keys()


class TrajectoryWriter:
    """Append-first writer for a ``*.mrec`` trajectory store.

    Wraps the Rust ``FrameSequenceWriter``. Frames are buffered and landed
    whole inner chunks at a time on a cadence derived from the frame size (or
    ``flush_every``); :meth:`flush` and :meth:`close` commit whatever is
    buffered, durably unless ``durable=False``. Use as a context manager, or
    call :meth:`close`.

    Args:
        path: Destination filesystem path for the store.
        schema: The :class:`SequenceSchema` every appended frame is checked
            against.
        flush_every: Land every this many frames instead of the derived cadence.
        compression: How floating-point columns are compressed: ``None``,
            ``"gzip[:level]"`` or ``"zstd[:level]"``. Integer, boolean and
            string columns always carry gzip level 1.
        durable: Whether :meth:`flush` / :meth:`close` fsync the touched files.
        meta: The record's identity document, written to ``meta/``.
    """

    def __init__(
        self,
        path: str | Path,
        schema: SequenceSchema,
        *,
        flush_every: int | None = None,
        compression: str | None = None,
        durable: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._inner = _MrecTrajectoryWriter(
            _path(path), schema._inner, flush_every, compression, durable, meta
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        flush_every: int | None = None,
        durable: bool = True,
    ) -> TrajectoryWriter:
        """Reattach to an existing store and continue appending after its last
        committed frame. Anything a crash left past the commit marker is rolled
        back first."""
        writer = cls.__new__(cls)
        writer._inner = _MrecTrajectoryWriter.open(_path(path), flush_every, durable)
        return writer

    def append(self, frame, step: int | None = None, time: float | None = None) -> None:
        """Buffer a frame. Omit ``step`` to number frames 0, 1, 2, …; pass it
        (with optional ``time`` in fs) for real MD numbering, or ``time`` alone
        to keep automatic numbering and still record times."""
        self._inner.append(frame, step, time)

    def flush(self) -> None:
        """Commit every buffered frame (durably, unless ``durable=False``)."""
        self._inner.flush()

    def close(self) -> None:
        """Commit and release the store. Idempotent."""
        self._inner.close()

    @property
    def flush_every(self) -> int:
        """The landing cadence in force, in frames."""
        return self._inner.flush_every

    @property
    def committed(self) -> int:
        """Frames committed so far."""
        return self._inner.committed

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def pack(path: str | Path) -> str:
    """Pack a closed ``*.mrec`` directory into a sibling ``*.mrec.zip`` and
    remove the directory. Every entry is stored, byte-identical to the file it
    replaces, so a :class:`TrajectoryReader` opens the archive with the same
    random access."""
    return _pack(_path(path))


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
    """Read the ``meta`` document of a ``*.mrec`` store (empty when the
    producer wrote none)."""
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
    "pack",
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
