"""Tests for the molpy-compatible trajectory readers in ``molrs.io``.

Self-contained fixtures written by molrs. No external corpus.
"""

from __future__ import annotations

import pytest

import molrs


class TestReturnsReaderNotList:
    def test_dcd_returns_reader(self, water_dcd):
        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        assert isinstance(reader, molrs.io.TrajectoryReader)
        assert reader.n_frames == len(reader) > 0

    def test_lammps_returns_reader(self, water_lammpstrj):
        reader = molrs.io.read_lammps_trajectory(str(water_lammpstrj))
        assert isinstance(reader, molrs.io.TrajectoryReader)
        assert reader.n_frames > 0

    def test_xyz_facade_returns_reader_but_toplevel_returns_list(self, water_xyz):
        path = str(water_xyz)
        reader = molrs.io.read_xyz_trajectory(path)
        assert isinstance(reader, molrs.io.TrajectoryReader)
        eager = molrs.io.raw.read_xyz_trajectory(path)
        assert isinstance(eager, list)
        assert reader.n_frames == len(eager)


class TestTrajectoryReaderSurface:
    def test_read_frame_and_negative_index(self, water_dcd):
        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        n = reader.n_frames
        assert reader.read_frame(0) is not None
        assert (
            reader.read_frame(-1)["atoms"].nrows
            == reader.read_frame(n - 1)["atoms"].nrows
        )

    def test_out_of_range_raises(self, water_dcd):
        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        with pytest.raises(IndexError):
            reader.read_frame(10_000_000)

    def test_read_all_and_read_range(self, water_dcd):
        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        n = reader.n_frames
        assert len(reader.read_all()) == n
        assert len(reader.read_range()) == n
        assert len(reader.read_range(0, n)) == n

    def test_read_range_step_zero_raises(self, water_dcd):
        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        with pytest.raises(ValueError):
            reader.read_range(0, 1, 0)

    def test_slicing(self, water_dcd):
        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        n = reader.n_frames
        assert isinstance(reader[0:1], list)
        assert len(reader[:]) == n
        assert len(reader[::-1]) == n

    def test_iteration(self, water_dcd):
        reader = molrs.io.read_dcd_trajectory(str(water_dcd))
        assert sum(1 for _ in reader) == reader.n_frames
        assert sum(1 for _ in reader) == reader.n_frames

    def test_context_manager_closes(self, water_dcd):
        with molrs.io.read_dcd_trajectory(str(water_dcd)) as reader:
            assert reader.n_frames > 0
        with pytest.raises(ValueError):
            reader.read_frame(0)


class TestMultiFile:
    def test_concatenates_frame_counts(self, water_dcd):
        path = str(water_dcd)
        single = molrs.io.read_dcd_trajectory(path).n_frames
        doubled = molrs.io.read_dcd_trajectory([path, path])
        assert doubled.n_frames == 2 * single

    def test_multi_file_indexing_crosses_boundary(self, water_dcd):
        path = str(water_dcd)
        single = molrs.io.read_dcd_trajectory(path).n_frames
        reader = molrs.io.read_dcd_trajectory([path, path])
        a = reader.read_frame(0)["atoms"].nrows
        b = reader.read_frame(single)["atoms"].nrows
        assert a == b
        assert len(reader.read_all()) == 2 * single


class TestCanonicalFields:
    def test_lammps_canonical_columns(self, water_lammpstrj):
        reader = molrs.io.read_lammps_trajectory(str(water_lammpstrj))
        atoms = reader.read_frame(0)["atoms"]
        assert "q" not in atoms
