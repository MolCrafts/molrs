"""The ``read_*_trajectory`` doors behind :mod:`molrs.io`.

Private: :mod:`molrs.io` is these functions' public path, and
:class:`~molrs.io.trajectory.TrajectoryReader` the object they return.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

from .._lib import DCDTrajReader as _DCDTrajReader
from .._lib import LAMMPSTrajReader as _LAMMPSTrajReader
from .._lib import TRRTrajReader as _TRRTrajReader
from .._lib import XTCTrajReader as _XTCTrajReader
from .._lib import XYZTrajReader as _XYZTrajReader
from .trajectory import TrajectoryReader

PathInput = str | PathLike[str]


def _as_paths(file: PathInput | Sequence[PathInput]) -> list[PathInput]:
    """Normalise a single path or a sequence of paths to a list."""
    if isinstance(file, (str, PathLike)):
        return [file]
    return list(file)


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
