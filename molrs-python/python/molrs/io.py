"""Python-shaped I/O facade for molrs.

This module exposes the Rust readers/writers under names that mirror the
``molpy.io`` API so molrs can serve as a drop-in replacement in tests
and downstream code. The Rust reader auto-detects the LAMMPS atom style
from column count and the ``Atoms # <style>`` section comment, so the
``atom_style`` keyword is accepted but not required.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

from .molrs import (
    read_lammps as _read_lammps,
    read_pdb as _read_pdb,
    read_xyz as _read_xyz,
    write_lammps as _write_lammps,
    write_pdb as _write_pdb,
    write_xyz as _write_xyz,
)


def read_lammps_data(
    file: str | PathLike[str],
    atom_style: str | None = None,
    frame: Any = None,
) -> Any:
    """Read a LAMMPS data file. molpy-compatible signature.

    Args:
        file: Path to the LAMMPS data file.
        atom_style: Accepted for API parity with molpy; the Rust reader
            auto-detects the style from the file (column count and the
            optional ``Atoms # <style>`` comment).
        frame: Reserved for API parity. molrs always returns a new
            ``Frame``; passing an existing frame is not supported.

    Returns:
        A molrs ``Frame`` whose ``box`` (alias for ``simbox``) and
        per-section blocks mirror the molpy layout for atoms, bonds,
        angles, dihedrals, and impropers.
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_lammps_data does not accept an existing frame; "
            "it always returns a new Frame."
        )
    return _read_lammps(str(file))


def read_pdb(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read a PDB file. molpy-compatible signature."""
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_pdb does not accept an existing frame."
        )
    return _read_pdb(str(file))


def read_xyz(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read an XYZ file. molpy-compatible signature."""
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_xyz does not accept an existing frame."
        )
    return _read_xyz(str(file))


def write_lammps_data(
    file: str | PathLike[str],
    frame: Any,
    atom_style: str | None = None,
) -> None:
    """Write a LAMMPS data file. molpy-compatible signature.

    ``atom_style`` is accepted for API parity but the writer derives the
    style from the columns present in ``frame['atoms']``.
    """
    _write_lammps(str(file), frame)


def write_pdb(file: str | PathLike[str], frame: Any) -> None:
    """Write a PDB file."""
    _write_pdb(str(file), frame)


def write_xyz(file: str | PathLike[str], frame: Any) -> None:
    """Write an XYZ file."""
    _write_xyz(str(file), frame)


__all__ = [
    "read_lammps_data",
    "read_pdb",
    "read_xyz",
    "write_lammps_data",
    "write_pdb",
    "write_xyz",
]
