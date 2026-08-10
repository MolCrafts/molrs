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

::: molrs.rotate

::: molrs.translate

::: molrs.scale

## I/O

::: molrs.io.read_pdb

::: molrs.io.read_xyz

::: molrs.io.read_xyz_trajectory

::: molrs.io.raw.read_lammps

::: molrs.io.raw.read_lammps_traj

::: molrs.io.raw.LAMMPSTrajReader

::: molrs.io.raw.read_dcd

::: molrs.io.raw.DCDTrajReader

::: molrs.io.raw.XYZTrajReader

::: molrs.io.read_gro

::: molrs.io.raw.read_chgcar_file

::: molrs.io.raw.read_cube_file

::: molrs.io.raw.write_cube_file

::: molrs.io.write_pdb

::: molrs.io.write_xyz

::: molrs.io.raw.write_lammps

::: molrs.io.raw.write_lammps_traj

## Regions and Neighbor Search

::: molrs.Sphere

::: molrs.HollowSphere

::: molrs.Cuboid

::: molrs.Region

::: molrs.NeighborList

::: molrs.Neighbors

::: molrs.NeighborQuery

## 3D Conformer Generation

::: molrs.conformer.Conformer

::: molrs.conformer.ConformerStageReport

::: molrs.conformer.ConformerReport

## Force Fields

The native force-field model exposes a `Style`/`Type` hierarchy
(`BondStyle`/`BondType`, `PairStyle`/`PairType`, …) and `Parameters`.

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

::: molrs.ff.Parameters

::: molrs.ff.MMFF94Typifier

::: molrs.ff.MMFF94STypifier

::: molrs.ff.OPLSAATypifier

::: molrs.ff.typifier.Typifier

::: molrs.ff.Potentials

::: molrs.optimize.LBFGS

::: molrs.optimize.OptReport

::: molrs.ff.read_forcefield_xml

::: molrs.ff.read_opls_xml

::: molrs.ff.extract_coords

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

Electrolyte transport kernels (ports of the *tame* recipes). See the
[Transport Kernels](../guides/transport.md) guide for signatures, units, and
worked examples.

### `molrs.compute.transport`

::: molrs.compute.transport.Onsager

::: molrs.compute.transport.Persist
