import os
import tempfile

import pytest

import molrs


class TestReadPdb:
    def test_basic(self, pdb_dir):
        frame = molrs.read_pdb(str(pdb_dir / "water.pdb"))
        assert "atoms" in frame
        assert frame["atoms"].nrows > 0

    def test_has_coordinates(self, pdb_dir):
        frame = molrs.read_pdb(str(pdb_dir / "water.pdb"))
        atoms = frame["atoms"]
        assert atoms.view("x") is not None
        assert atoms.view("y") is not None
        assert atoms.view("z") is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_pdb("/nonexistent/path.pdb")


class TestReadGro:
    def test_native_basic(self, gro_dir):
        frames = molrs.read_gro(str(gro_dir / "ubiquitin.gro"))
        assert len(frames) == 1
        f0 = frames[0]
        assert "atoms" in f0
        assert f0["atoms"].nrows > 0
        assert f0.box is not None

    def test_native_columns(self, gro_dir):
        frames = molrs.read_gro(str(gro_dir / "ubiquitin.gro"))
        atoms = frames[0]["atoms"]
        for col in ["resid", "resname", "atom_name", "atom_id", "x", "y", "z"]:
            assert col in atoms, f"missing column: {col}"

    def test_facade_canonical_columns(self, gro_dir):
        frames = molrs.io.read_gro(str(gro_dir / "ubiquitin.gro"))
        atoms = frames[0]["atoms"]
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing canonical column: {col}"

    def test_facade_no_format_native_columns(self, gro_dir):
        frames = molrs.io.read_gro(str(gro_dir / "ubiquitin.gro"))
        atoms = frames[0]["atoms"]
        for col in ["resid", "atom_name", "atom_id"]:
            assert col not in atoms, f"format-native column leaked: {col}"

    def test_round_trip(self, gro_dir):
        frames = molrs.io.read_gro(str(gro_dir / "ubiquitin.gro"))
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
            molrs.read_gro("/nonexistent/path.gro")

    def test_triclinic_box(self, gro_dir):
        frames = molrs.io.read_gro(str(gro_dir / "1vln-triclinic.gro"))
        assert frames[0].box is not None


class TestReadXyz:
    def test_basic(self, xyz_dir):
        frame = molrs.read_xyz(str(xyz_dir / "methane.xyz"))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 5

    def test_has_coordinates(self, xyz_dir):
        frame = molrs.read_xyz(str(xyz_dir / "methane.xyz"))
        atoms = frame["atoms"]
        assert atoms.view("x") is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_xyz("/nonexistent/path.xyz")
