"""Shared pytest fixtures for the molrs Python bindings.

Binding tests are **self-contained**: they never require third-party scientific
software (RDKit, AmberTools, freud, …) and never require the optional
chemfiles-derived ``tests-data/`` corpus. Format smoke tests build tiny
frames in-process and round-trip them through molrs writers/readers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import molrs


@pytest.fixture
def cubic_box():
    """A 10x10x10 cubic box."""
    return molrs.Box.cube(10.0)


@pytest.fixture
def ortho_box():
    """A 5x10x15 orthorhombic box."""
    return molrs.Box.ortho(np.array([5.0, 10.0, 15.0], dtype=np.float64))


@pytest.fixture
def sample_points():
    """5 points inside a 10x10x10 box."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [9.0, 8.0, 7.0],
            [0.1, 0.1, 0.1],
            [9.9, 9.9, 9.9],
        ],
        dtype=np.float64,
    )


def _water_frame(*, for_lammps: bool = False) -> molrs.Frame:
    """Minimal 3-atom frame with box.

    Keep columns minimal so writers (esp. extended XYZ) do not emit fields the
    matching reader cannot parse. LAMMPS data needs ``type`` (+ optional charge).
    """
    f = molrs.Frame()
    b = molrs.Block()
    b.insert("symbol", ["O", "H", "H"])
    b.insert("x", np.array([0.0, 0.96, -0.24], dtype=np.float64))
    b.insert("y", np.array([0.0, 0.0, 0.93], dtype=np.float64))
    b.insert("z", np.array([0.0, 0.0, 0.0], dtype=np.float64))
    b.insert("name", ["O", "H1", "H2"])
    b.insert("res_name", ["SOL", "SOL", "SOL"])
    b.insert("res_id", np.array([1, 1, 1], dtype=np.int32))
    b.insert("id", np.array([1, 2, 3], dtype=np.int32))
    if for_lammps:
        b.insert("type", np.array([1, 2, 2], dtype=np.int32))
        b.insert("charge", np.array([-0.834, 0.417, 0.417], dtype=np.float64))
        b.insert("mol_id", np.array([1, 1, 1], dtype=np.int32))
    f["atoms"] = b
    f.box = molrs.Box.cube(10.0)
    return f


@pytest.fixture
def water_frame() -> molrs.Frame:
    """Fresh water-like frame (callers may mutate)."""
    return _water_frame()


@pytest.fixture
def water_xyz(tmp_path: Path) -> Path:
    path = tmp_path / "water.xyz"
    molrs.write_xyz(str(path), _water_frame())
    return path


@pytest.fixture
def water_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "water.pdb"
    molrs.write_pdb(str(path), _water_frame())
    return path


@pytest.fixture
def water_gro(tmp_path: Path) -> Path:
    path = tmp_path / "water.gro"
    molrs.write_gro(str(path), _water_frame())
    return path


@pytest.fixture
def water_dcd(tmp_path: Path) -> Path:
    path = tmp_path / "water.dcd"
    frame = _water_frame()
    molrs.write_dcd(str(path), [frame, frame])
    return path


@pytest.fixture
def water_trr(tmp_path: Path) -> Path:
    path = tmp_path / "water.trr"
    frame = _water_frame()
    molrs.write_trr(str(path), [frame, frame])
    return path


@pytest.fixture
def water_xtc(tmp_path: Path) -> Path:
    path = tmp_path / "water.xtc"
    frame = _water_frame()
    molrs.write_xtc(str(path), [frame, frame])
    return path


@pytest.fixture
def water_lammpstrj(tmp_path: Path) -> Path:
    path = tmp_path / "water.lammpstrj"
    frame = _water_frame(for_lammps=True)
    molrs.write_lammps_traj(str(path), [frame, frame])
    return path


@pytest.fixture
def water_lammps_data(tmp_path: Path) -> Path:
    path = tmp_path / "water.data"
    molrs.write_lammps(str(path), _water_frame(for_lammps=True))
    return path
