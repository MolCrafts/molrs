# Python Reference

```python
import molrs
```

The top level of `molrs` is its subsystems and nothing else, exactly as the
Rust crate's root is: every symbol has one path, the Python module named after
its Rust owner (`molrs.core.Frame` is `molrs::core::Frame`). Import the
subsystem you use:

```python
from molrs.core import Frame, Block
from molrs.core import Box, NeighborList
from molrs.ff.potential import PotentialCompiler
```

This page is rendered from the installed `molrs` package by
`mkdocstrings-python`. The type stub `molrs-python/python/molrs/_lib.pyi` is
the committed companion artifact that keeps signatures visible to static tools
and the docs build.

| Module | Rust owner | Holds |
|---|---|---|
| `molrs.core` | `molrs::core` | `Block`, `Frame`, `FrameMeta`, `MetaValue`, `MetaDocument`, `Trajectory`, `ScalarObservable`, `VectorObservable`, `BlockDtypeError`; `Box` (Rust `SimBox`), `NeighborList`, `Neighbors`, `NeighborQuery`, `VerletSkin`, the regions, `TriMesh`, `Trace`; `MolGraph`, `Atomistic`, `CoarseGrain`, the node / relation views, `ExtractedSubgraph`, `Element`, `Topology`; `Unit`, `Quantity`, `UnitRegistry`, `UnitPreset`, `UnitsError` |
| `molrs.core.keys` | `molrs::core::keys` | the canonical column, frame-meta and graph keys |
| `molrs.core.schema` | `molrs::core::schema` | `ColumnSpec`, `BlockSpec`, the block names, `relation_endpoints` |
| `molrs.core.constants` | `molrs::core::constants` | every physical and engine constant (`AVOGADRO`, `COULOMB_REAL`, `AMBER_COULOMB`, `AMBER_SCEE`, …) |
| `molrs.op` | `molrs::op` | `superpose`, `centroid`, `Fit`, `DEFAULT_GAP_TOL` |
| `molrs.perceive` | `molrs::perceive` | `Perceive`, `RingInfo`, `SmartsPattern`, `SmartsMatch`, `Reaction`, `SubgraphMatcher` |
| `molrs.io` | `molrs::io` | every file reader and writer, as a function `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]` (`_str` / `_bytes` in memory): structure, trajectory and force-field files, `*.mrec` records (`read_mrec_frame` / `write_mrec_frame` and partners), wire-encoded frames, SMILES and CGsmiles text, the LAMMPS log, CSV blocks |
| `molrs.io.pdb`, `.xyz`, `.gro`, `.dcd`, `.trr`, `.xtc` | `molrs::io::{pdb, xyz, gro, dcd, trr, xtc}` | each format's lazy reader: `PdbReader`, `XyzReader`, `GroReader`, `DcdReader`, `TrrReader`, `XtcReader` |
| `molrs.io.lammps` | `molrs::io::lammps` | `LammpsDumpReader`, `BondReactTemplate`, the `Lammps*` log records |
| `molrs.io.smiles` | `molrs::io::smiles` | `SmilesIR`, `SmilesError`, `BondingDescriptor` |
| `molrs.io.cgsmiles` | `molrs::io::cgsmiles` | `CGSmilesIR` and the CGsmiles records |
| `molrs.io.mrec` | `molrs::io::mrec` | `MOLREC_VERSION`, `RESERVED_META_KEYS`, `MrecReader`, `MrecWriter`, `SequenceSchema`, `ForceFieldSection`, `section_names`, `pack_mrec_zip`, `validation` |
| `molrs.ff.forcefield` | `molrs::ff::forcefield` | `ForceField`, the `Style` / `Type` handles (the data model; its files are `molrs.io`'s) |
| `molrs.ff.potential` | `molrs::ff::potential` | `PotentialCompiler`, `Potentials`, `TypedPotentials`, `kernel`, `LJCut`, `intramolecular_pairs`, `Potential` |
| `molrs.ff.typifier` | `molrs::ff::typifier` | `Typifier`, `Match`, the built-in typifiers, `assign_cmaps` |
| `molrs.ff.charge` | `molrs::ff::charge` | `BccModel`, `MullikenModel`, `GasteigerModel` |
| `molrs.ff.ir` | `molrs::ff::ir` | the force-field IR registry and its `IrError` family |
| `molrs.ff.params` | `molrs::ff::params` | `clpol_polarizability` |
| `molrs.ff.scale_lj` | `molrs::ff::scale_lj` | `FragmentScaling`, `compute_k_ij`, `fragment_scaling_data`, `scale_lj` |
| `molrs.optimize` | `molrs::optimize` | `LBFGS`, `OptReport` |
| `molrs.md` | `molrs::md` | `VelocityVerlet`, `Langevin`, `MDState`, `MaxwellBoltzmann`, `MD` |
| `molrs.conformer` | `molrs::conformer` | `Conformer`, `ConformerReport`, `ConformerStageReport` |
| `molrs.builder` | `molrs::builder` | `GrapheneBuilder`, `CarbonTubeBuilder`, `Assembler` and its placers / orienter, `Coarsener` |
| `molrs.compute` | `molrs::compute` | every analysis, flat, and the `Compute` protocol |
| `molrs.signal` | `molrs::signal` | `acf_fft`, `xcorr_fft`, `apply_window`, `frequency_grid` |
| `molrs.stream` | `molrs::stream` | `Publisher`, `ControlCommand` |

## `molrs.core`

::: molrs.core.Block

::: molrs.core.Frame

::: molrs.core.FrameMeta

::: molrs.core.MetaValue

Every door of `frame.meta` hands back a frozen value: a fixed-length vector
is a `tuple`, and a JSON object is a `MetaDocument`. Nested arrays are
tuples. `json.dumps` accepts a tuple and rejects a document — use
`json.dumps(frame.meta["run"].copy())`. Order inside a nested document is
unspecified.

::: molrs.core.MetaDocument

::: molrs.core.Trajectory

::: molrs.core.ScalarObservable

::: molrs.core.VectorObservable

::: molrs.core.Atomistic

::: molrs.core.CoarseGrain

::: molrs.core.MolGraph

Rigid-body moves are methods of `Atomistic` and `CoarseGrain`, not module
functions: `translate(delta)`, `rotate(axis, angle, about=None)` and
`scale(factor, about=None)`. Each moves every node that has coordinates in
place and returns the graph itself, so moves chain:
`mol.translate([1, 0, 0]).rotate([0, 0, 1], 0.5).scale([2, 2, 2])`.
In Rust the moves are `molrs::op::geometry`'s functions over a `MolGraph`.

::: molrs.core.Box

A region is a solid with a signed distance to its boundary: every class
answers `contains`, `distance` (negative inside) and `bounds`, and composes
with `&`, `|` and `~`. Outside a shape is `~shape`; a shell is
`outer & ~inner`. `TriMesh` is the surface a `Polyhedron` is bounded by and
what `molrs.io.read_stl` reads (the WASM binding reads the same file with
`readStlBytes` into `Mesh`).

::: molrs.core.Sphere

::: molrs.core.Cuboid

::: molrs.core.Parallelepiped

::: molrs.core.HalfSpace

::: molrs.core.Cylinder

::: molrs.core.Ellipsoid

::: molrs.core.TriMesh

::: molrs.core.Polyhedron

::: molrs.core.SphereUnion

::: molrs.core.Region

::: molrs.core.NeighborList

::: molrs.core.Neighbors

::: molrs.core.NeighborQuery

## `molrs.perceive`

::: molrs.perceive.Perceive

::: molrs.perceive.RingInfo

::: molrs.perceive.Reaction

## `molrs.io`

One module per file format. Every door is a function named after its format:
`read_X` / `write_X` for one frame, `read_X_trajectory` /
`write_X_trajectory` for a sequence, `_str` / `_bytes` for text and bytes in
memory; family formats carry the family name (`read_lammps_data`,
`read_amber_prmtop`, `read_vasp_poscar`). No door picks the format for the
caller. Every reader emits the canonical column names (`molrs.core.keys`).
Each `read_X_trajectory` returns its format's lazy reader,
`molrs.io.X.<X>Reader`, over one path or a list of paths.

### Structure files

::: molrs.io.read_pdb

::: molrs.io.write_pdb

::: molrs.io.read_xyz

::: molrs.io.write_xyz

::: molrs.io.read_gro

::: molrs.io.write_gro

::: molrs.io.read_sdf

::: molrs.io.read_mol2

::: molrs.io.write_mol2

::: molrs.io.read_cif

::: molrs.io.write_cif

::: molrs.io.read_xsf

::: molrs.io.write_xsf

::: molrs.io.read_cube

::: molrs.io.write_cube

::: molrs.io.read_vasp_poscar

::: molrs.io.write_vasp_poscar

::: molrs.io.read_vasp_chgcar

::: molrs.io.read_stl

### Trajectories

::: molrs.io.read_pdb_trajectory

::: molrs.io.write_pdb_trajectory

::: molrs.io.pdb.PdbReader

::: molrs.io.read_xyz_trajectory

::: molrs.io.write_xyz_trajectory

::: molrs.io.xyz.XyzReader

::: molrs.io.read_gro_trajectory

::: molrs.io.write_gro_trajectory

::: molrs.io.gro.GroReader

::: molrs.io.read_dcd_trajectory

::: molrs.io.write_dcd_trajectory

::: molrs.io.dcd.DcdReader

::: molrs.io.read_trr_trajectory

::: molrs.io.write_trr_trajectory

::: molrs.io.trr.TrrReader

::: molrs.io.read_xtc_trajectory

::: molrs.io.write_xtc_trajectory

::: molrs.io.xtc.XtcReader

### LAMMPS (`molrs.io.lammps`)

::: molrs.io.read_lammps_data

::: molrs.io.write_lammps_data

::: molrs.io.read_lammps_molecule

::: molrs.io.write_lammps_molecule

::: molrs.io.read_lammps_molecule_json

::: molrs.io.write_lammps_molecule_json

::: molrs.io.read_lammps_trajectory

::: molrs.io.write_lammps_trajectory

::: molrs.io.write_lammps_dump_local

::: molrs.io.lammps.LammpsDumpReader

::: molrs.io.lammps.BondReactTemplate

::: molrs.io.write_lammps_bond_react_map

::: molrs.io.write_lammps_bond_react_system

::: molrs.io.read_lammps_log

::: molrs.io.read_lammps_log_str

### AMBER

::: molrs.io.read_amber_prmtop

::: molrs.io.read_amber_inpcrd

::: molrs.io.read_amber_ac

::: molrs.io.read_amber_prep

::: molrs.io.write_amber_prep

### Record files (`molrs.io.mrec`)

The [Record files guide](../guides/records.md) shows these in use.

::: molrs.io.read_mrec_frame

::: molrs.io.write_mrec_frame

::: molrs.io.read_mrec_system

::: molrs.io.write_mrec_system

::: molrs.io.read_mrec_trajectory

::: molrs.io.write_mrec_trajectory

::: molrs.io.read_mrec_forcefield

::: molrs.io.write_mrec_forcefield

::: molrs.io.mrec.section_names

::: molrs.io.read_mrec_meta

::: molrs.io.mrec.SequenceSchema

::: molrs.io.mrec.MrecWriter

::: molrs.io.mrec.MrecReader

::: molrs.io.mrec.ForceFieldSection

::: molrs.io.mrec.pack_mrec_zip

### Force-field files

::: molrs.io.read_lammps_forcefield

::: molrs.io.write_lammps_forcefield

::: molrs.io.read_gromacs_top_forcefield

::: molrs.io.write_gromacs_top_forcefield

::: molrs.io.read_gromacs_system

::: molrs.io.write_gromacs_system

::: molrs.io.read_amber_prmtop_forcefield

::: molrs.io.read_amber_prmtop_system

::: molrs.io.write_amber_frcmod

::: molrs.io.read_openmm_xml_forcefield

::: molrs.io.write_openmm_xml_forcefield

::: molrs.io.read_molrs_xml_forcefield

::: molrs.io.write_molrs_xml_forcefield

### SMILES and CGsmiles

::: molrs.io.read_smiles_str

::: molrs.io.write_smiles_str

::: molrs.io.smiles.SmilesIR

::: molrs.io.smiles.SmilesError

::: molrs.io.read_cgsmiles_str

::: molrs.io.cgsmiles.CGSmilesIR

### Wire-encoded frames and CSV blocks

::: molrs.io.read_msgpack_frame_bytes

::: molrs.io.write_msgpack_frame_bytes

::: molrs.io.read_json_frame_str

::: molrs.io.write_json_frame_str

::: molrs.io.read_csv_block

::: molrs.io.read_csv_block_str

::: molrs.io.write_csv_block

::: molrs.io.write_csv_block_str

## `molrs.ff`

### `molrs.ff.forcefield`

The native force-field model exposes a `Style`/`Type` handle hierarchy
(`BondStyle`/`BondType`, `PairStyle`/`PairType`, `CmapStyle`/`CmapType`,
…); a handle's `params` is a plain dict of numbers, strings and float64
arrays (a CMAP `grid`).

::: molrs.ff.forcefield.ForceField

::: molrs.ff.forcefield.Style

::: molrs.ff.forcefield.AtomStyle

::: molrs.ff.forcefield.BondStyle

::: molrs.ff.forcefield.AngleStyle

::: molrs.ff.forcefield.DihedralStyle

::: molrs.ff.forcefield.ImproperStyle

::: molrs.ff.forcefield.PairStyle

::: molrs.ff.forcefield.CmapStyle

::: molrs.ff.forcefield.Type

::: molrs.ff.forcefield.AtomType

::: molrs.ff.forcefield.BondType

::: molrs.ff.forcefield.AngleType

::: molrs.ff.forcefield.DihedralType

::: molrs.ff.forcefield.ImproperType

::: molrs.ff.forcefield.PairType

::: molrs.ff.forcefield.CmapType

### `molrs.ff.potential`

::: molrs.ff.potential.PotentialCompiler

::: molrs.ff.potential.Potentials

### `molrs.ff.typifier`

::: molrs.ff.typifier.MMFF94Typifier

::: molrs.ff.typifier.MMFF94STypifier

::: molrs.ff.typifier.OPLSAATypifier

::: molrs.ff.typifier.AtdTypifier

::: molrs.ff.typifier.GaffTypifier

::: molrs.ff.typifier.Typifier

::: molrs.ff.typifier.Match

### `molrs.ff.charge`

::: molrs.ff.charge.GasteigerModel

### `molrs.ff.params`

::: molrs.ff.params.clpol_polarizability

## `molrs.optimize`

::: molrs.optimize.LBFGS

::: molrs.optimize.OptReport

## `molrs.conformer`

::: molrs.conformer.Conformer

::: molrs.conformer.ConformerStageReport

::: molrs.conformer.ConformerReport

## `molrs.builder`

::: molrs.builder.Coarsener

## `molrs.compute`

The Rust compute facade is flat, and so is `molrs.compute`: every analysis is
`molrs.compute.<Name>`.

### Structure

::: molrs.compute.RDF

::: molrs.compute.RDFResult

::: molrs.compute.GaussianDensity

::: molrs.compute.LocalDensity

::: molrs.compute.StaticStructureFactorDebye

::: molrs.compute.PMFTXY

::: molrs.compute.BondOrientationalOrder

### Order

::: molrs.compute.Steinhardt

::: molrs.compute.Nematic

::: molrs.compute.Hexatic

::: molrs.compute.SolidLiquid

### Clusters and shape

::: molrs.compute.Cluster

::: molrs.compute.ClusterResult

::: molrs.compute.ClusterCenters

::: molrs.compute.ClusterCentersResult

::: molrs.compute.ClusterProperties

::: molrs.compute.CenterOfMass

::: molrs.compute.CenterOfMassResult

::: molrs.compute.GyrationTensor

::: molrs.compute.InertiaTensor

::: molrs.compute.RadiusOfGyration

### Dynamics

::: molrs.compute.MSD

::: molrs.compute.MSDResult

::: molrs.compute.MSDTimeSeries

### Descriptors

::: molrs.compute.DescriptorRow

::: molrs.compute.Pca2

::: molrs.compute.PcaResult

::: molrs.compute.KMeans

::: molrs.compute.KMeansResult

### Transport

Electrolyte transport kernels (ports of the *tame* recipes). Worked examples,
units, and signatures are in the
[molpy documentation](https://docs.molcrafts.org/molpy/).

::: molrs.compute.Onsager

::: molrs.compute.Persist
