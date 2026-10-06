"""Format-native I/O bindings — one-to-one with ``molrs::io``.

These are the compiled readers/writers exactly as Rust exposes them: columns
keep their **format-native** names (``resid``, ``q``, ``symbol``), because that
is what the file said. :mod:`molrs.io` wraps each of them with a
:class:`~molrs.fields.FieldFormatter` and is what callers normally want.

Reach for this module when the format-native spelling *is* the thing under
test — the FFI seam, a parser edge case, a column a formatter would rename.

Names pair as in :mod:`molrs.io` (``read_X`` / ``write_X``,
``read_X_trajectory`` / ``write_X_trajectory``); every trajectory reader here
returns an eager ``list[Frame]``, and the ``*TrajReader`` classes are the lazy
cursors.
"""

from __future__ import annotations

from .._lib import (
    DCDTrajReader,
    LAMMPSTrajReader,
    TRRTrajReader,
    XTCTrajReader,
    XYZTrajReader,
    parse_lammps_log_text,
    read_amber_inpcrd,
    read_amber_prmtop,
    read_chgcar,
    read_cube,
    read_dcd_trajectory,
    read_gro,
    read_gro_trajectory,
    read_lammps_data,
    read_lammps_log,
    read_lammps_molecule,
    read_lammps_trajectory,
    read_mol2,
    read_pdb,
    read_pdb_trajectory,
    read_trr_trajectory,
    read_xsf,
    read_xtc_trajectory,
    read_xyz,
    read_xyz_trajectory,
    write_cube,
    write_dcd_trajectory,
    write_gro,
    write_gro_trajectory,
    write_lammps_data,
    write_lammps_dump_local,
    write_lammps_molecule,
    write_lammps_trajectory,
    write_mol2,
    write_pdb,
    write_pdb_trajectory,
    write_trr_trajectory,
    write_xsf,
    write_xtc_trajectory,
    write_xyz,
    write_xyz_trajectory,
)

__all__ = [
    "DCDTrajReader",
    "LAMMPSTrajReader",
    "TRRTrajReader",
    "XTCTrajReader",
    "XYZTrajReader",
    "parse_lammps_log_text",
    "read_amber_inpcrd",
    "read_amber_prmtop",
    "read_chgcar",
    "read_cube",
    "read_dcd_trajectory",
    "read_gro",
    "read_gro_trajectory",
    "read_lammps_data",
    "read_lammps_log",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    "read_mol2",
    "read_pdb",
    "read_pdb_trajectory",
    "read_trr_trajectory",
    "read_xsf",
    "read_xtc_trajectory",
    "read_xyz",
    "read_xyz_trajectory",
    "write_cube",
    "write_dcd_trajectory",
    "write_gro",
    "write_gro_trajectory",
    "write_lammps_data",
    "write_lammps_dump_local",
    "write_lammps_molecule",
    "write_lammps_trajectory",
    "write_mol2",
    "write_pdb",
    "write_pdb_trajectory",
    "write_trr_trajectory",
    "write_xsf",
    "write_xtc_trajectory",
    "write_xyz",
    "write_xyz_trajectory",
]
