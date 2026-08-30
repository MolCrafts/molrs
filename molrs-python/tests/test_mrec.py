"""FFI smoke tests for the ``molrs.io.mrec`` path doors.

Record and Trajectory are memory carriers; the on-disk doors live only in
``molrs.io.mrec``. Depth (layout, version rejection) lives in the Rust unit
tests. This file proves the Python seam: lazy frame 0, system+meta round-trip,
deleted carrier methods, and that ``molrs.io.TrajectoryReader`` stays the dump
concatenator.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import molrs

# Å; dyadic so a bit-exact f64 round-trip is the golden, not a tolerance.
_N_ATOMS = 3
_ATOM_X = (0.0, 1.0, 0.5)
_ATOM_Y = (0.25, 0.0, 2.0)
_ATOM_Z = (0.0, 4.0, 0.125)


def _coords_frame() -> molrs.Frame:
    atoms = molrs.Block()
    atoms["x"] = np.array(_ATOM_X, dtype=np.float64)
    atoms["y"] = np.array(_ATOM_Y, dtype=np.float64)
    atoms["z"] = np.array(_ATOM_Z, dtype=np.float64)
    frame = molrs.Frame()
    frame["atoms"] = atoms
    return frame


def _assert_coords(frame: molrs.Frame) -> None:
    atoms = frame["atoms"]
    assert atoms.nrows == _N_ATOMS
    np.testing.assert_array_equal(np.asarray(atoms["x"]), np.array(_ATOM_X, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(atoms["y"]), np.array(_ATOM_Y, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(atoms["z"]), np.array(_ATOM_Z, dtype=np.float64))


class TestTrajectoryReader:
    def test_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        trajectory = molrs.Trajectory([_coords_frame()])
        molrs.io.mrec.write_trajectory(str(path), trajectory)

        reader = molrs.io.mrec.TrajectoryReader(str(path))
        frame = reader.read_frame(0)
        _assert_coords(frame)


class TestRecordDoors:
    def test_write_record(self, tmp_path: Path) -> None:
        path = tmp_path / "record.mrec"
        record = molrs.Record()
        record.set_system(_coords_frame())
        record.meta = {"creator": {"name": "pytest"}}
        molrs.io.mrec.write_record(str(path), record)

        loaded = molrs.io.mrec.read_record(str(path))
        assert loaded.system is not None
        _assert_coords(loaded.system)
        assert loaded.meta["creator"]["name"] == "pytest"

    def test_read_record(self, tmp_path: Path) -> None:
        path = tmp_path / "record.mrec"
        record = molrs.Record()
        record.set_system(_coords_frame())
        record.meta = {"creator": {"name": "pytest"}}
        molrs.io.mrec.write_record(str(path), record)

        loaded = molrs.io.mrec.read_record(str(path))
        assert loaded.meta["format_name"] == "mrec"
        assert loaded.meta["record_schema_version"] == 1
        assert loaded.system is not None
        _assert_coords(loaded.system)


class TestRemovedCarrierDoors:
    def test_record_read_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            molrs.Record.read

    def test_record_write_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            molrs.Record.write

    def test_trajectory_read_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            molrs.Trajectory.read

    def test_trajectory_write_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            molrs.Trajectory.write


class TestMrecSurface:
    def test_from_molrs_io_mrec_import_trajectory_reader(self) -> None:
        from molrs.io.mrec import TrajectoryReader

        assert TrajectoryReader is molrs.io.mrec.TrajectoryReader

    def test_io_import_yields_dump_concatenator(self, water_dcd: Path) -> None:
        from molrs.io import TrajectoryReader

        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        assert isinstance(reader, TrajectoryReader)
        assert reader.n_frames == 2
        assert reader.read_frame(0)["atoms"].nrows == 3
        params = inspect.signature(TrajectoryReader.__init__).parameters
        assert "readers" in params
        assert "path" not in params

    def test_mrec_trajectory_reader_is_not_the_dump_class(self) -> None:
        from molrs.io import TrajectoryReader as DumpReader
        from molrs.io.mrec import TrajectoryReader as MrecReader

        assert MrecReader is not DumpReader
        params = inspect.signature(MrecReader.__init__).parameters
        assert "path" in params

    def test_no_frame_reader(self) -> None:
        assert hasattr(molrs.io.mrec, "FrameReader") is False

    def test_no_make_helpers(self) -> None:
        makers = [name for name in dir(molrs.io.mrec) if name.startswith("make_")]
        assert makers == []

    def test_no_god_reader(self) -> None:
        assert not hasattr(molrs.io.mrec, "Reader")
