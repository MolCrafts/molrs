"""FFI smoke tests for the ``molrs.io.mrec`` path doors.

Frame and Trajectory are the in-memory objects; the on-disk doors live only in
``molrs.io.mrec``. Schema checks live in ``molrs::io::mrec::schema`` and are
bound, not reimplemented, at ``molrs.io.mrec.schema``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import molrs
import numpy as np
import pytest

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
    np.testing.assert_array_equal(
        np.asarray(atoms["x"]), np.array(_ATOM_X, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["y"]), np.array(_ATOM_Y, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["z"]), np.array(_ATOM_Z, dtype=np.float64)
    )


class TestTrajectoryReader:
    def test_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        trajectory = molrs.Trajectory([_coords_frame()])
        molrs.io.mrec.write_trajectory(path, trajectory)

        reader = molrs.io.mrec.TrajectoryReader(path)
        frame = reader.read_frame(0)
        _assert_coords(frame)


class TestFrameDoors:
    def test_write_and_read_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        molrs.io.mrec.write_frame(path, _coords_frame())
        _assert_coords(molrs.io.mrec.read_frame(path))
        assert molrs.io.mrec.sections(path) == frozenset({"meta", "frame"})
        meta = molrs.io.mrec.read_meta(path)
        molrs.io.mrec.schema.validate_meta(meta)
        # Development contract: no version key is stamped; the document is
        # exactly what the producer handed in (nothing here).
        assert meta == {}

    def test_write_frame_with_system(self, tmp_path: Path) -> None:
        path = tmp_path / "both.mrec"
        molrs.io.mrec.write_frame(path, _coords_frame(), system=_coords_frame())
        _assert_coords(molrs.io.mrec.read_frame(path))
        _assert_coords(molrs.io.mrec.read_system(path))
        assert molrs.io.mrec.sections(path) == frozenset({"meta", "frame", "system"})


class TestSystemDoors:
    def test_write_and_read_system(self, tmp_path: Path) -> None:
        path = tmp_path / "system.mrec"
        molrs.io.mrec.write_system(path, _coords_frame())
        _assert_coords(molrs.io.mrec.read_system(path))
        assert "frame" not in molrs.io.mrec.sections(path)


class TestTrajectoryDoors:
    def test_write_and_read_trajectory(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        molrs.io.mrec.write_trajectory(path, molrs.Trajectory([_coords_frame()]))
        loaded = molrs.io.mrec.read_trajectory(path)
        assert len(loaded) == 1
        _assert_coords(loaded[0])


class TestRemovedCarrierDoors:
    def test_record_is_not_public(self) -> None:
        assert not hasattr(molrs, "Record")
        assert not hasattr(molrs, "MolRec")
        assert not hasattr(molrs, "Observables")

    def test_record_read_write_are_gone(self) -> None:
        assert not hasattr(molrs.io.mrec, "read_record")
        assert not hasattr(molrs.io.mrec, "write_record")

    def test_trajectory_read_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            molrs.Trajectory.read

    def test_trajectory_write_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            molrs.Trajectory.write


class TestSchema:
    def test_version_constant_comes_from_molrs(self) -> None:
        assert molrs.io.mrec.schema.MOLREC_VERSION == 1
        assert molrs.io.mrec.schema.MOLREC_VERSION == molrs._lib.MREC_MOLREC_VERSION
        assert molrs.io.mrec.schema.RESERVED_META_KEYS == ["molrec_version"]

    def test_a_missing_molrec_version_is_not_validated(self) -> None:
        # Development contract: absent means no version check; the retired
        # brand keys are neither required nor refused.
        molrs.io.mrec.schema.validate_meta(
            {"record_schema_version": 1, "format_name": "mrec"}
        )
        molrs.io.mrec.schema.validate_meta({})

    def test_a_present_molrec_version_out_of_range_is_refused(self) -> None:
        with pytest.raises(Exception, match="molrec_version"):
            molrs.io.mrec.schema.validate_meta({"molrec_version": 0})
        with pytest.raises(Exception, match="molrec_version"):
            molrs.io.mrec.schema.validate_meta({"molrec_version": "1"})
        molrs.io.mrec.schema.validate_meta({"molrec_version": 1})

    def test_retired_path_is_refused(self) -> None:
        with pytest.raises(Exception, match="\\.mrec"):
            molrs.io.mrec.schema.validate_path("water.zarr")

    def test_empty_frame_passes(self) -> None:
        molrs.io.mrec.schema.validate_frame(molrs.Frame())


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
