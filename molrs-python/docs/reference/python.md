# Python Reference

```python
import molrs
```

The top level of `molrs` is its subsystems and nothing else, exactly as the
Rust crate's root is: every symbol has one path, the Python module named after
its Rust owner (`molrs.store.Frame` is `molrs::store::Frame`). Import the
subsystem you use:

```python
from molrs.store import Frame, Block
from molrs.spatial import Box, NeighborList
from molrs.ff.potential import PotentialCompiler
```

This page is rendered from the installed `molrs` package by
`mkdocstrings-python`. The type stub `molrs-python/python/molrs/_lib.pyi` is
the committed companion artifact that keeps signatures visible to static tools
and the docs build.

| Module | Rust owner | Holds |
|---|---|---|
| `molrs.store` | `molrs::store` | `Block`, `Frame`, `FrameMeta`, `MetaValue`, `MetaDocument`, `Trajectory`, `ScalarObservable`, `VectorObservable`, `BlockDtypeError`; `keys`, `schema` |
| `molrs.spatial` | `molrs::spatial` | `Box`, `NeighborList`, `Neighbors`, `NeighborQuery`, `VerletSkin`, the regions, `TriMesh`, `Trace` |
| `molrs.system` | `molrs::system` | `Graph`, `Atomistic`, `CoarseGrain`, the node / relation views, `ExtractedSubgraph`, `Element`, `Topology` |
| `molrs.units` | `molrs::units` | `Unit`, `Quantity`, `UnitRegistry`, `UnitPreset`, `UnitsError`, `AMBER_COULOMB` |
| `molrs.op` | `molrs::op` | `superpose`, `centroid`, `Fit`, `DEFAULT_GAP_TOL` |
| `molrs.perceive` | `molrs::perceive` | `Perceive`, `RingInfo`, `SmartsPattern`, `SmartsMatch`, `Reaction`, `SubgraphMatcher` |
| `molrs.io` | `molrs::io` | structure and trajectory readers / writers, `TrajectoryReader`, `SmilesIR`, `CGSmilesIR`, `SmilesError`, the LAMMPS log, `*.mrec` doors |
| `molrs.io.mrec` | `molrs::io::mrec` | `SequenceSchema`, `TrajectoryWriter`, `TrajectoryReader`, `ForceFieldSection`, `pack`, `schema` |
| `molrs.ff.forcefield` | `molrs::ff::forcefield` | `ForceField`, the `Style` / `Type` handles, the force-field file readers and writers |
| `molrs.ff.potential` | `molrs::ff::potential` | `PotentialCompiler`, `Potentials`, `TypedPotentials`, `kernel`, `LJCut`, `intramolecular_pairs`, `Potential` |
| `molrs.ff.typifier` | `molrs::ff::typifier` | `Typifier`, `Match`, the built-in typifiers, `assign_cmaps` |
| `molrs.ff.charge` | `molrs::ff::charge` | `BccModel`, `MullikenModel`, `GasteigerModel` |
| `molrs.ff.ir` | `molrs::ff::ir` | the force-field IR registry and its `IrError` family |
| `molrs.ff.params` | `molrs::ff::params` | `AMBER_SCEE`, `AMBER_SCNB`, `clpol_polarizability` |
| `molrs.ff.scale_lj` | `molrs::ff::scale_lj` | `FragmentScaling`, `compute_k_ij`, `fragment_scaling_data`, `scale_lj` |
| `molrs.optimize` | `molrs::optimize` | `LBFGS`, `OptReport` |
| `molrs.md` | `molrs::md` | `VelocityVerlet`, `Langevin`, `MDState`, `MaxwellBoltzmann`, `MD` |
| `molrs.conformer` | `molrs::conformer` | `Conformer`, `ConformerReport`, `ConformerStageReport` |
| `molrs.builder` | `molrs::builder` | `GrapheneBuilder`, `CarbonTubeBuilder`, `Assembler` and its placers / orienter, `Coarsener` |
| `molrs.compute` | `molrs::compute` | every analysis, flat, and the `Compute` protocol |
| `molrs.signal` | `molrs::signal` | `acf_fft`, `xcorr_fft`, `apply_window`, `frequency_grid` |
| `molrs.stream` | `molrs::stream` | `Publisher`, `ControlCommand`, `read_frame_bytes`, `write_frame_bytes` |

## `molrs.store`

::: molrs.store.Block

::: molrs.store.Frame

::: molrs.store.FrameMeta

::: molrs.store.MetaValue

Every door of `frame.meta` hands back a frozen value: a fixed-length vector
is a `tuple`, and a JSON object is a `MetaDocument`. Nested arrays are
tuples. `json.dumps` accepts a tuple and rejects a document — use
`json.dumps(frame.meta["run"].copy())`. Order inside a nested document is
unspecified.

::: molrs.store.MetaDocument

::: molrs.store.Trajectory

::: molrs.store.ScalarObservable

::: molrs.store.VectorObservable

## `molrs.system`

::: molrs.system.Atomistic

::: molrs.system.CoarseGrain

::: molrs.system.Graph

Rigid-body moves are methods of `Atomistic` and `CoarseGrain`, not module
functions: `translate(delta)`, `rotate(axis, angle, about=None)` and
`scale(factor, about=None)`. Each moves every node that has coordinates in
place and returns the graph itself, so moves chain:
`mol.translate([1, 0, 0]).rotate([0, 0, 1], 0.5).scale([2, 2, 2])`.

## `molrs.spatial`

::: molrs.spatial.Box

A region is a solid with a signed distance to its boundary: every class
answers `contains`, `distance` (negative inside) and `bounds`, and composes
with `&`, `|` and `~`. Outside a shape is `~shape`; a shell is
`outer & ~inner`. `TriMesh` is the surface a `Polyhedron` is bounded by and
what `molrs.io.read_stl` reads (the WASM binding reads the same file with
`readSTL` into `Mesh`).

::: molrs.spatial.Sphere

::: molrs.spatial.Cuboid

::: molrs.spatial.Parallelepiped

::: molrs.spatial.HalfSpace

::: molrs.spatial.Cylinder

::: molrs.spatial.Ellipsoid

::: molrs.spatial.TriMesh

::: molrs.spatial.Polyhedron

::: molrs.spatial.SphereUnion

::: molrs.spatial.Region

::: molrs.spatial.NeighborList

::: molrs.spatial.Neighbors

::: molrs.spatial.NeighborQuery

## `molrs.perceive`

::: molrs.perceive.Perceive

::: molrs.perceive.RingInfo

::: molrs.perceive.Reaction

## `molrs.io`

Reader and writer names pair: `read_X` / `write_X` for one frame,
`read_X_trajectory` / `write_X_trajectory` for a sequence. Every reader emits
the canonical column names (`molrs.store.keys`). `read_frame` /
`write_frame` pick the format from the file name. The LAMMPS dump, XYZ, DCD,
TRR and XTC trajectory readers return a lazy `TrajectoryReader`.

::: molrs.io.read_frame

::: molrs.io.write_frame

::: molrs.io.TrajectoryReader

::: molrs.io.read_pdb

::: molrs.io.write_pdb

::: molrs.io.read_pdb_trajectory

::: molrs.io.write_pdb_trajectory

::: molrs.io.read_xyz

::: molrs.io.write_xyz

::: molrs.io.read_xyz_trajectory

::: molrs.io.write_xyz_trajectory

::: molrs.io.read_gro

::: molrs.io.write_gro

::: molrs.io.read_gro_trajectory

::: molrs.io.write_gro_trajectory

::: molrs.io.read_lammps_data

::: molrs.io.write_lammps_data

::: molrs.io.BondReactTemplate

::: molrs.io.write_bond_react_map

::: molrs.io.write_lammps_bond_react_system

::: molrs.io.read_lammps_trajectory

::: molrs.io.write_lammps_trajectory

::: molrs.io.write_lammps_dump_local

::: molrs.io.read_dcd_trajectory

::: molrs.io.write_dcd_trajectory

::: molrs.io.read_trr_trajectory

::: molrs.io.write_trr_trajectory

::: molrs.io.read_xtc_trajectory

::: molrs.io.write_xtc_trajectory

::: molrs.io.SmilesIR

::: molrs.io.read_mrec

::: molrs.io.write_mrec

::: molrs.io.read_mrec_system

::: molrs.io.write_mrec_system

::: molrs.io.read_mrec_trajectory

::: molrs.io.write_mrec_trajectory

::: molrs.io.read_mrec_forcefield

::: molrs.io.write_mrec_forcefield

::: molrs.io.mrec_sections

::: molrs.io.read_mrec_meta

### Record files (`molrs.io.mrec`)

The [Record files guide](../guides/records.md) shows these in use.

::: molrs.io.mrec.SequenceSchema

::: molrs.io.mrec.TrajectoryWriter

::: molrs.io.mrec.TrajectoryReader

::: molrs.io.mrec.ForceFieldSection

::: molrs.io.mrec.pack

### Other formats

::: molrs.io.read_chgcar

::: molrs.io.read_cube

::: molrs.io.write_cube

::: molrs.io.read_mol2

::: molrs.io.write_mol2

::: molrs.io.read_amber_inpcrd

::: molrs.io.read_stl

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

::: molrs.ff.forcefield.read_forcefield_xml

::: molrs.ff.forcefield.read_opls_xml

::: molrs.ff.forcefield.write_gromacs_system

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

::: molrs.compute.BondOrder

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
