"""FFI smoke tests for top-level IO readers/writers.

Self-contained: every fixture is written by molrs itself. No external corpus,
no third-party scientific software.
"""

from __future__ import annotations

import os
import tempfile

import pytest

import molrs


class TestErrorMessages:
    def test_pyo3_type_error_names_the_argument(self):
        with pytest.raises(TypeError, match="center"):
            molrs.Sphere("not-an-array", 1.0)


class TestReadPdb:
    def test_basic(self, water_pdb):
        frame = molrs.io.raw.read_pdb(str(water_pdb))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self, water_pdb):
        frame = molrs.io.raw.read_pdb(str(water_pdb))
        atoms = frame["atoms"]
        assert atoms.view("x") is not None
        assert atoms.view("y") is not None
        assert atoms.view("z") is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.raw.read_pdb("/nonexistent/path.pdb")

    def test_missing_file_names_the_path(self):
        with pytest.raises(OSError, match="missing.pdb"):
            molrs.io.read_pdb("missing.pdb")


class TestReadGro:
    def test_native_basic(self, water_gro):
        frames = molrs.io.raw.read_gro(str(water_gro))
        assert len(frames) == 1
        f0 = frames[0]
        assert "atoms" in f0
        assert f0["atoms"].nrows == 3
        assert f0.box is not None

    def test_native_columns(self, water_gro):
        frames = molrs.io.raw.read_gro(str(water_gro))
        atoms = frames[0]["atoms"]
        # The reader emits canonical names directly; `resid`/`atom_id` were
        # format-native spellings that something downstream had to rename, and
        # that rename is now a write into a UInt key an Int column cannot pass.
        for col in ["res_id", "resname", "atom_name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing column: {col}"

    def test_facade_canonical_columns(self, water_gro):
        frames = molrs.io.read_gro(str(water_gro))
        atoms = frames[0]["atoms"]
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing canonical column: {col}"

    def test_facade_no_format_native_columns(self, water_gro):
        frames = molrs.io.read_gro(str(water_gro))
        atoms = frames[0]["atoms"]
        for col in ["resid", "atom_name", "atom_id"]:
            assert col not in atoms, f"format-native column leaked: {col}"

    def test_round_trip(self, water_gro):
        frames = molrs.io.read_gro(str(water_gro))
        f0 = frames[0]
        with tempfile.NamedTemporaryFile(suffix=".gro", delete=False) as tmp:
            tmpname = tmp.name
        try:
            molrs.io.write_gro(tmpname, f0)
            frames2 = molrs.io.read_gro(tmpname)
            f1 = frames2[0]
            assert f0["atoms"].nrows == f1["atoms"].nrows
        finally:
            os.unlink(tmpname)

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.raw.read_gro("/nonexistent/path.gro")


class TestReadXyz:
    def test_basic(self, water_xyz):
        frame = molrs.io.raw.read_xyz(str(water_xyz))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self, water_xyz):
        frame = molrs.io.raw.read_xyz(str(water_xyz))
        atoms = frame["atoms"]
        assert atoms.view("x") is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.raw.read_xyz("/nonexistent/path.xyz")
