"""Tests for the molpy-compatible trajectory readers in ``molrs.io``.

Self-contained fixtures written by molrs. No external corpus.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest


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
        eager = molrs.io.read_xyz_trajectory(path).read_all()
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


class TestDumpLocalWrite:
    def test_write_bonds_roundtrip(self, tmp_path):
        atoms = molrs.store.Block()
        atoms["id"] = np.array([1, 2, 3], dtype=np.uint64)
        atoms["x"] = np.array([0.0, 1.0, 2.0])
        atoms["y"] = np.zeros(3)
        atoms["z"] = np.zeros(3)
        bonds = molrs.store.Block()
        bonds["atomi"] = np.array([0, 1], dtype=np.uint64)
        bonds["atomj"] = np.array([1, 2], dtype=np.uint64)
        frame = molrs.store.Frame()
        frame["atoms"] = atoms
        frame["bonds"] = bonds
        frame.box = molrs.spatial.Box.cube(10.0)
        path = tmp_path / "bonds.dump.local"
        molrs.io.write_lammps_dump_local(path, [frame])
        text = path.read_text()
        assert "ITEM: NUMBER OF ENTRIES" in text
        assert "batom1 batom2" in text
        loaded = molrs.io.read_lammps_trajectory(str(path)).read_all()
        assert loaded[0]["entries"].nrows == 2


class TestDumpColumnChoice:
    @staticmethod
    def _frame():
        atoms = molrs.store.Block()
        atoms["id"] = np.array([1, 2], dtype=np.uint64)
        atoms["mol_id"] = np.array([1, 1], dtype=np.uint64)
        atoms["mass"] = np.array([16.0, 1.008])
        atoms["element"] = ["O", "H"]
        atoms["x"] = np.array([0.0, 1.0])
        atoms["y"] = np.array([0.0, 2.0])
        atoms["z"] = np.array([0.0, 3.0])
        frame = molrs.store.Frame()
        frame["atoms"] = atoms
        frame.box = molrs.spatial.Box.cube(10.0)
        return frame

    def test_writes_only_the_named_columns_in_order(self, tmp_path):
        path = tmp_path / "chosen.lammpstrj"
        molrs.io.write_lammps_trajectory(
            path, [self._frame()], columns=["id", "element", "mol", "x", "y", "z"]
        )
        text = path.read_text()
        assert "ITEM: ATOMS id element mol x y z" in text
        assert "mass" not in text

    def test_default_writes_every_column(self, tmp_path):
        path = tmp_path / "all.lammpstrj"
        molrs.io.write_lammps_trajectory(path, [self._frame()])
        assert "ITEM: ATOMS id element mass mol x y z" in path.read_text()

    def test_rejects_a_column_the_frame_lacks(self, tmp_path):
        path = tmp_path / "missing.lammpstrj"
        with pytest.raises(OSError, match="'q'"):
            molrs.io.write_lammps_trajectory(path, [self._frame()], columns=["id", "q"])


class TestDumpTypeField:
    @staticmethod
    def _atoms(**extra):
        return {
            "id": [1, 2, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.5, 0.0],
            "z": [0.0, 0.0, 0.25],
            **extra,
        }

    def test_string_type_labels_round_trip(self, tmp_path):
        path = tmp_path / "labels.lammpstrj"
        frame = molrs.store.Frame(
            {"atoms": self._atoms(type=["OW", "HW", "HW"])}, box=molrs.spatial.Box.cube(10.0)
        )
        molrs.io.write_lammps_trajectory(path, [frame])
        text = path.read_text()
        assert "ITEM: ATOMS id type x y z\n1 OW " in text
        atoms = molrs.io.read_lammps_trajectory(str(path)).read_all()[0]["atoms"]
        assert list(atoms["type"]) == ["OW", "HW", "HW"]
        assert "type_id" not in atoms

    def test_type_id_wins_the_type_field(self, tmp_path):
        path = tmp_path / "both.lammpstrj"
        frame = molrs.store.Frame(
            {"atoms": self._atoms(type=["OW", "HW", "HW"], type_id=[1, 2, 2])},
            box=molrs.spatial.Box.cube(10.0),
        )
        molrs.io.write_lammps_trajectory(path, [frame])
        text = path.read_text()
        assert "ITEM: ATOMS id type x y z\n1 1 " in text
        atoms = molrs.io.read_lammps_trajectory(str(path)).read_all()[0]["atoms"]
        assert list(atoms["type_id"]) == [1, 2, 2]
        assert "type" not in atoms


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
