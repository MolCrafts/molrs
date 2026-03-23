import os
import pytest
import numpy as np
import molrs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestReadPdb:
    def test_basic(self):
        frame = molrs.read_pdb(os.path.join(DATA_DIR, "water.pdb"))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self):
        frame = molrs.read_pdb(os.path.join(DATA_DIR, "water.pdb"))
        atoms = frame["atoms"]
        x = atoms.view("x")
        y = atoms.view("y")
        z = atoms.view("z")
        assert x is not None
        assert y is not None
        assert z is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_pdb("/nonexistent/path.pdb")


class TestReadXyz:
    def test_basic(self):
        frame = molrs.read_xyz(os.path.join(DATA_DIR, "water.xyz"))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self):
        frame = molrs.read_xyz(os.path.join(DATA_DIR, "water.xyz"))
        atoms = frame["atoms"]
        x = atoms.view("x")
        assert x is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_xyz("/nonexistent/path.xyz")


class TestRoundtrip:
    def test_pdb_to_target(self):
        """Full workflow: read PDB → create Target."""
        frame = molrs.read_pdb(os.path.join(DATA_DIR, "water.pdb"))
        target = molrs.Target(frame, 5)
        assert target.natoms == 3
        assert target.count == 5
