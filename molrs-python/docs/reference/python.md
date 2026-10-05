# Python Reference

Canonical import style:

```python
import molrs as mr
```

This page is rendered from the installed `molrs` package by `mkdocstrings-python`.
Autodoc identifiers use the package name (`molrs.Frame`); user code should
`import molrs as mr` and write `mr.Frame`. The type stub in
`molrs-python/python/molrs/_lib.pyi` is the committed companion artifact that
keeps signatures visible to static tools and the docs build.

## Core Model

::: molrs.Box

::: molrs.Block

::: molrs.Frame

::: molrs.FrameMeta

::: molrs.MetaValue

Every door of `frame.meta` hands back a frozen value: a fixed-length vector
is a `tuple`, and a JSON object is a `MetaDocument`. Nested arrays are
tuples. `json.dumps` accepts a tuple and rejects a document — use
`json.dumps(frame.meta["run"].copy())`. Order inside a nested document is
unspecified.

::: molrs.MetaDocument

## Topology and SMILES

::: molrs.Atomistic

::: molrs.CoarseGrain

::: molrs.Graph

::: molrs.io.SmilesIR

## Chemistry Perception

::: molrs.perceive.Perceive

::: molrs.perceive.RingInfo

::: molrs.ff.charge.GasteigerModel

## Transforms

Rigid-body moves are methods of `Atomistic` and `CoarseGrain`, not module
functions: `translate(delta)`, `rotate(axis, angle, about=None)` and
`scale(factor, about=None)`. Each moves every node that has coordinates in
place and returns the graph itself, so moves chain:
`mol.translate([1, 0, 0]).rotate([0, 0, 1], 0.5).scale([2, 2, 2])`.

## I/O

Reader and writer names pair: `read_X` / `write_X` for one frame,
`read_X_trajectory` / `write_X_trajectory` for a sequence. `molrs.io` returns
canonical field names; `molrs.io.raw` keeps the format-native ones and reads
trajectories eagerly.

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

::: molrs.io.read_lammps_trajectory

::: molrs.io.write_lammps_trajectory

::: molrs.io.write_lammps_dump_local

::: molrs.io.read_dcd_trajectory

::: molrs.io.write_dcd_trajectory

::: molrs.io.read_trr_trajectory

::: molrs.io.write_trr_trajectory

::: molrs.io.read_xtc_trajectory

::: molrs.io.write_xtc_trajectory

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

::: molrs.io.read_chgcar

::: molrs.io.read_cube

::: molrs.io.write_cube

::: molrs.io.raw.LAMMPSTrajReader

::: molrs.io.raw.DCDTrajReader

::: molrs.io.raw.XYZTrajReader

## Regions and Neighbor Search

A region is a solid with a signed distance to its boundary: every class
answers `contains`, `distance` (negative inside) and `bounds`, and composes
with `&`, `|` and `~`. Outside a shape is `~shape`; a shell is
`outer & ~inner`. `TriMesh` is the surface a `Polyhedron` is bounded by and
what `molrs.io.read_stl` reads (the WASM binding reads the same file with
`readSTL` into `Mesh`).

::: molrs.Sphere

::: molrs.Cuboid

::: molrs.Parallelepiped

::: molrs.HalfSpace

::: molrs.Cylinder

::: molrs.Ellipsoid

::: molrs.TriMesh

::: molrs.Polyhedron

::: molrs.SphereUnion

::: molrs.Region

::: molrs.io.read_stl

::: molrs.NeighborList

::: molrs.Neighbors

::: molrs.NeighborQuery

## 3D Conformer Generation

::: molrs.conformer.Conformer

::: molrs.conformer.ConformerStageReport

::: molrs.conformer.ConformerReport

## Force Fields

The native force-field model exposes a `Style`/`Type` handle hierarchy
(`BondStyle`/`BondType`, `PairStyle`/`PairType`, …); a handle's `params`
is a plain dict.

::: molrs.ff.ForceField

::: molrs.ff.Style

::: molrs.ff.AtomStyle

::: molrs.ff.BondStyle

::: molrs.ff.AngleStyle

::: molrs.ff.DihedralStyle

::: molrs.ff.ImproperStyle

::: molrs.ff.PairStyle

::: molrs.ff.Type

::: molrs.ff.AtomType

::: molrs.ff.BondType

::: molrs.ff.AngleType

::: molrs.ff.DihedralType

::: molrs.ff.ImproperType

::: molrs.ff.PairType

::: molrs.ff.MMFF94Typifier

::: molrs.ff.MMFF94STypifier

::: molrs.ff.OPLSAATypifier

::: molrs.ff.typifier.Typifier

::: molrs.ff.typifier.Match

::: molrs.ff.PotentialCompiler

::: molrs.ff.Potentials

::: molrs.optimize.LBFGS

::: molrs.optimize.OptReport

::: molrs.ff.read_forcefield_xml

::: molrs.ff.read_opls_xml

## Trajectory

::: molrs.Trajectory

::: molrs.ScalarObservable

::: molrs.VectorObservable

## Analysis

Analysis classes live under the `molrs.compute` subpackage, organized by
domain. The layout mirrors freud and the underlying Rust crate
(`molrs_compute::{density, order, environment, …}`).

### `molrs.compute.density`

::: molrs.compute.density.RDF

::: molrs.compute.density.RDFResult

::: molrs.compute.density.GaussianDensity

::: molrs.compute.density.LocalDensity

### `molrs.compute.order`

::: molrs.compute.order.Steinhardt

::: molrs.compute.order.Nematic

::: molrs.compute.order.Hexatic

::: molrs.compute.order.SolidLiquid

### `molrs.compute.environment`

::: molrs.compute.environment.BondOrder

### `molrs.compute.pmft`

::: molrs.compute.pmft.PMFTXY

### `molrs.compute.diffraction`

::: molrs.compute.diffraction.StaticStructureFactorDebye

### `molrs.compute.cluster`

::: molrs.compute.cluster.Cluster

::: molrs.compute.cluster.ClusterResult

::: molrs.compute.cluster.ClusterCenters

::: molrs.compute.cluster.ClusterCentersResult

::: molrs.compute.cluster.ClusterProperties

::: molrs.compute.cluster.CenterOfMass

::: molrs.compute.cluster.CenterOfMassResult

::: molrs.compute.cluster.GyrationTensor

::: molrs.compute.cluster.InertiaTensor

::: molrs.compute.cluster.RadiusOfGyration

### `molrs.compute.msd`

::: molrs.compute.msd.MSD

::: molrs.compute.msd.MSDResult

::: molrs.compute.msd.MSDTimeSeries

### `molrs.compute.ml`

::: molrs.compute.ml.DescriptorRow

::: molrs.compute.ml.Pca2

::: molrs.compute.ml.PcaResult

::: molrs.compute.ml.KMeans

::: molrs.compute.ml.KMeansResult

## Transport

Electrolyte transport kernels (ports of the *tame* recipes). Worked examples,
units, and signatures are in the
[molpy documentation](https://docs.molcrafts.org/molpy/).

### `molrs.compute.transport`

::: molrs.compute.transport.Onsager

::: molrs.compute.transport.Persist
