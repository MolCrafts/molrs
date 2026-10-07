"""File I/O — ``molrs::io``. Every file reader and writer is here.

Structure, trajectory and force-field files, SMILES text, ``*.mrec`` records
and frame bytes alike: a factory that reads or writes a format has exactly one
shape —

* a function at the top of this module, ``read_<fmt>[_<what>]`` /
  ``write_<fmt>[_<what>]``; or
* a class of the format's own submodule, ``molrs.io.<fmt>.<Fmt>Reader`` /
  ``<Fmt>Writer`` (:class:`molrs.io.trajectory.TrajectoryReader`,
  :class:`molrs.io.mrec.MrecReader` / :class:`~molrs.io.mrec.MrecWriter`).

A class that belongs to one format lives in that format's submodule:
:mod:`molrs.io.smiles` (:class:`~molrs.io.smiles.SmilesIR`, the CGsmiles
records, :class:`~molrs.io.smiles.SmilesError`), :mod:`molrs.io.log` (the
LAMMPS log records), :mod:`molrs.io.lammps_bond_react`
(:class:`~molrs.io.lammps_bond_react.BondReactTemplate`), :mod:`molrs.io.mrec`
(the record store's reader, writer, schema and force-field section) and
:mod:`molrs.io.trajectory`.

Every structure reader emits the project-wide canonical column names
(:mod:`molrs.core.keys`: ``element``, ``res_id``, ``charge``, ``mol_id``,
…) — the Rust readers map a format's own spelling (``symbol``, ``resSeq``,
``q``, ``mol``) at the boundary — and every writer takes them.

Names come in pairs. A format ``X`` that holds one frame is read by
``read_X`` and written by ``write_X``; a sequence of frames is read by
``read_X_trajectory`` and written by ``write_X_trajectory`` — ``lammps`` dumps,
``xyz``, ``pdb``, ``gro``, ``dcd``, ``trr`` and ``xtc``. A format
with one direction only (``read_chgcar``, ``write_lammps_dump_local``, …) has
no partner by nature, not by omission. A reader of in-memory text rather than
a path carries ``_str`` (``read_lammps_log_str``,
``write_lammps_forcefield_str``).

``read_lammps_trajectory``, ``read_xyz_trajectory``, ``read_dcd_trajectory``,
``read_trr_trajectory`` and ``read_xtc_trajectory`` return a lazy
:class:`~molrs.io.trajectory.TrajectoryReader` rather than a ``list[Frame]``;
each accepts a single path or a list of paths (frames are concatenated) and
yields canonical field names. ``read_pdb_trajectory`` and
``read_gro_trajectory`` read the whole file into a ``list[Frame]`` (no seekable
native reader backs them). Every path argument takes a ``str`` or any
``os.PathLike``.

Force-field files map onto :class:`molrs.ff.forcefield.ForceField`, the data
model :mod:`molrs.ff.forcefield` owns:

* readers — :func:`read_lammps_forcefield`, :func:`read_lammps_data_coeffs`,
  :func:`read_lammps_cmap`, :func:`read_gromacs_top_ff`,
  :func:`read_gromacs_system`, :func:`read_amber_prmtop_ff`,
  :func:`read_amber_prmtop_system`, :func:`read_forcefield_xml`,
  :func:`read_opls_xml`
* writers — :func:`write_lammps_forcefield`,
  :func:`write_lammps_forcefield_str`, :func:`write_lammps_data_coeffs`,
  :func:`write_lammps_cmap`, :func:`write_gromacs_top_ff`,
  :func:`write_gromacs_system`, :func:`write_amber_frcmod`,
  :func:`write_forcefield_xml`

Each reader maps a format onto the force-field IR (adopts the LAMMPS
standard), units and factors included; each writer is the inverse.

A ``*.mrec`` scientific record is read and written whole by
:func:`read_mrec` / :func:`write_mrec` (Structure) and their ``_system`` /
``_trajectory`` / ``_forcefield`` partners, plus :func:`read_mrec_meta`; a run
too large for memory goes through :class:`molrs.io.mrec.MrecReader` /
:class:`~molrs.io.mrec.MrecWriter`.

:func:`read_frame_bytes` / :func:`write_frame_bytes` read and write a frame in
the wire encoding :class:`molrs.stream.Publisher` streams (``"msgpack"`` or
``"json"``).

:func:`read_smiles` reads one molecule from a SMILES string — connectivity
only, no implicit hydrogens added, no coordinates; a ``'.'``-separated set is
refused (take it apart with ``SmilesIR(s).components()``).
"""

from .._lib import (
    read_ac,
    read_amber_inpcrd,
    read_amber_prmtop,
    read_amber_prmtop_ff,
    read_amber_prmtop_system,
    read_chgcar,
    read_cube,
    read_forcefield_xml,
    read_frame,
    read_frame_bytes,
    read_gro,
    read_gro_trajectory,
    read_gromacs_system,
    read_gromacs_top_ff,
    read_lammps_cmap,
    read_lammps_data,
    read_lammps_data_coeffs,
    read_lammps_forcefield,
    read_lammps_log,
    read_lammps_log_str,
    read_lammps_molecule,
    read_mol2,
    read_mrec,
    read_mrec_forcefield,
    read_mrec_meta,
    read_mrec_system,
    read_mrec_trajectory,
    read_opls_xml,
    read_pdb,
    read_pdb_trajectory,
    read_prep,
    read_smiles,
    read_stl,
    read_xsf,
    read_xyz,
    write_amber_frcmod,
    write_bond_react_map,
    write_cube,
    write_dcd_trajectory,
    write_forcefield_xml,
    write_frame,
    write_frame_bytes,
    write_gro,
    write_gro_trajectory,
    write_gromacs_system,
    write_gromacs_top_ff,
    write_lammps_bond_react_system,
    write_lammps_cmap,
    write_lammps_data,
    write_lammps_data_coeffs,
    write_lammps_dump_local,
    write_lammps_forcefield,
    write_lammps_forcefield_str,
    write_lammps_molecule,
    write_lammps_trajectory,
    write_mol2,
    write_mrec,
    write_mrec_forcefield,
    write_mrec_system,
    write_mrec_trajectory,
    write_pdb,
    write_pdb_trajectory,
    write_prep,
    write_smarts,
    write_trr_trajectory,
    write_xsf,
    write_xtc_trajectory,
    write_xyz,
    write_xyz_trajectory,
)
from . import lammps_bond_react, log, mrec, smiles, trajectory
from ._csv import read_block_csv, write_block_csv
from ._trajectory_doors import (
    read_dcd_trajectory,
    read_lammps_trajectory,
    read_trr_trajectory,
    read_xtc_trajectory,
    read_xyz_trajectory,
)

# Defined in the private modules above; this module is their public path.
for _defined in (
    read_block_csv,
    read_dcd_trajectory,
    read_lammps_trajectory,
    read_trr_trajectory,
    read_xtc_trajectory,
    read_xyz_trajectory,
    write_block_csv,
):
    _defined.__module__ = __name__

__all__ = [
    "lammps_bond_react",
    "log",
    "mrec",
    "read_ac",
    "read_amber_inpcrd",
    "read_amber_prmtop",
    "read_amber_prmtop_ff",
    "read_amber_prmtop_system",
    "read_block_csv",
    "read_chgcar",
    "read_cube",
    "read_dcd_trajectory",
    "read_forcefield_xml",
    "read_frame",
    "read_frame_bytes",
    "read_gro",
    "read_gro_trajectory",
    "read_gromacs_system",
    "read_gromacs_top_ff",
    "read_lammps_cmap",
    "read_lammps_data",
    "read_lammps_data_coeffs",
    "read_lammps_forcefield",
    "read_lammps_log",
    "read_lammps_log_str",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    "read_mol2",
    "read_mrec",
    "read_mrec_forcefield",
    "read_mrec_meta",
    "read_mrec_system",
    "read_mrec_trajectory",
    "read_opls_xml",
    "read_pdb",
    "read_pdb_trajectory",
    "read_prep",
    "read_smiles",
    "read_stl",
    "read_trr_trajectory",
    "read_xsf",
    "read_xtc_trajectory",
    "read_xyz",
    "read_xyz_trajectory",
    "smiles",
    "trajectory",
    "write_amber_frcmod",
    "write_block_csv",
    "write_bond_react_map",
    "write_cube",
    "write_dcd_trajectory",
    "write_forcefield_xml",
    "write_frame",
    "write_frame_bytes",
    "write_gro",
    "write_gro_trajectory",
    "write_gromacs_system",
    "write_gromacs_top_ff",
    "write_lammps_bond_react_system",
    "write_lammps_cmap",
    "write_lammps_data",
    "write_lammps_data_coeffs",
    "write_lammps_dump_local",
    "write_lammps_forcefield",
    "write_lammps_forcefield_str",
    "write_lammps_molecule",
    "write_lammps_trajectory",
    "write_mol2",
    "write_mrec",
    "write_mrec_forcefield",
    "write_mrec_system",
    "write_mrec_trajectory",
    "write_pdb",
    "write_pdb_trajectory",
    "write_prep",
    "write_smarts",
    "write_trr_trajectory",
    "write_xsf",
    "write_xtc_trajectory",
    "write_xyz",
    "write_xyz_trajectory",
]
