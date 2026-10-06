"""``molrs.io.read_frame`` / ``write_frame``: one door that picks the format
from the file name (or a format name) and hands off to that format."""

from __future__ import annotations

import molrs
import numpy as np
import pytest


def _water() -> molrs.store.Frame:
    frame = molrs.store.Frame()
    atoms = molrs.store.Block()
    atoms.insert("element", ["O", "H", "H"])
    atoms.insert("x", np.array([0.0, 0.96, -0.24]))
    atoms.insert("y", np.array([0.0, 0.0, 0.93]))
    atoms.insert("z", np.zeros(3))
    frame["atoms"] = atoms
    frame.box = molrs.spatial.Box.cube(10.0)
    return frame


@pytest.mark.parametrize("name", ["w.pdb", "w.xyz", "w.mol2", "w.lammpstrj", "w.gro"])
def test_round_trip_by_extension(tmp_path, name):
    path = tmp_path / name
    molrs.io.write_frame(path, _water())
    back = molrs.io.read_frame(path)
    assert back["atoms"].nrows == 3
    np.testing.assert_allclose(back["atoms"]["x"], [0.0, 0.96, -0.24], atol=1e-3)


def test_an_explicit_format_overrides_the_name(tmp_path):
    path = tmp_path / "structure.txt"
    molrs.io.write_frame(path, _water(), format="xyz")
    assert molrs.io.read_frame(path, format="XYZ")["atoms"].nrows == 3
    with pytest.raises(OSError, match="format"):
        molrs.io.read_frame(path)


def test_unknown_and_read_only_formats_are_refused(tmp_path):
    with pytest.raises(OSError, match="unknown structure format"):
        molrs.io.read_frame(tmp_path / "w.xyz", format="docx")
    with pytest.raises(OSError, match="no sdf writer"):
        molrs.io.write_frame(tmp_path / "w.sdf", _water())
