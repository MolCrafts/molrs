"""Format-native I/O bindings — one-to-one with ``molrs::io``.

These are the compiled readers/writers exactly as Rust exposes them: columns
keep their **format-native** names (``resid``, ``q``, ``symbol``), because that
is what the file said. :mod:`molrs.io` wraps each of them with a
:class:`~molrs.fields.FieldFormatter` and is what callers normally want.

Reach for this module when the format-native spelling *is* the thing under
test — the FFI seam, a parser edge case, a column a formatter would rename.
"""

from __future__ import annotations

from .._lib import (
    DCDTrajReader as DCDTrajReader,
    LAMMPSTrajReader as LAMMPSTrajReader,
    TRRTrajReader as TRRTrajReader,
    XTCTrajReader as XTCTrajReader,
    XYZTrajReader as XYZTrajReader,
    read_chgcar_file as read_chgcar_file,
    read_cube_file as read_cube_file,
    read_dcd as read_dcd,
    read_gro as read_gro,
    read_lammps as read_lammps,
    read_lammps_traj as read_lammps_traj,
    read_pdb as read_pdb,
    read_pdb_trajectory as read_pdb_trajectory,
    read_trr as read_trr,
    read_xtc as read_xtc,
    read_xyz as read_xyz,
    read_xyz_trajectory as read_xyz_trajectory,
    write_cube_file as write_cube_file,
    write_dcd as write_dcd,
    write_gro as write_gro,
    write_lammps as write_lammps,
    write_lammps_traj as write_lammps_traj,
    write_pdb as write_pdb,
    write_pdb_trajectory as write_pdb_trajectory,
    write_trr as write_trr,
    write_xtc as write_xtc,
    write_xyz as write_xyz,
)

__all__ = [
    "read_pdb",
    "read_pdb_trajectory",
    "read_xyz",
    "read_xyz_trajectory",
    "XYZTrajReader",
    "read_lammps",
    "read_lammps_traj",
    "LAMMPSTrajReader",
    "read_dcd",
    "DCDTrajReader",
    "read_trr",
    "TRRTrajReader",
    "read_xtc",
    "XTCTrajReader",
    "read_gro",
    "read_chgcar_file",
    "read_cube_file",
    "write_cube_file",
    "write_pdb",
    "write_pdb_trajectory",
    "write_xyz",
    "write_gro",
    "write_lammps",
    "write_lammps_traj",
    "write_dcd",
    "write_trr",
    "write_xtc",
]
