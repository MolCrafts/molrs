"""Scientific-record I/O for ``*.mrec`` stores.

``Record`` and ``Trajectory`` are in-memory carriers. Path doors live here:

* :func:`read_record` / :func:`write_record` — whole record
* :func:`write_trajectory` — trajectory-only record
* :class:`TrajectoryReader` — lazy frame cursor wrapping Rust
  ``FrameSequence`` (one frame per :meth:`TrajectoryReader.read_frame`)

These names are not re-exported from :mod:`molrs.io`. That module's
:class:`~molrs.io.TrajectoryReader` remains the LAMMPS/XYZ/DCD dump
concatenator. There is no ``FrameReader``.
"""

from __future__ import annotations

from .._lib import (
    MrecTrajectoryReader as _MrecTrajectoryReader,
    read_record as read_record,
    write_record as write_record,
    write_trajectory as write_trajectory,
)


class TrajectoryReader:
    """Lazy one-frame cursor over a ``*.mrec`` trajectory.

    Wraps the Rust ``FrameSequence`` store cursor: construction opens the
    index, and :meth:`read_frame` decodes exactly the asked-for frame.

    Args:
        path: Filesystem path of the record store.
    """

    def __init__(self, path: str) -> None:
        self._inner = _MrecTrajectoryReader(path)

    def read_frame(self, index: int):
        """Decode one committed frame.

        Args:
            index: Zero-based frame index.

        Returns:
            The frame at ``index``.
        """
        return self._inner.read_frame(index)


__all__ = ["TrajectoryReader", "read_record", "write_record", "write_trajectory"]
