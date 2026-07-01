---
title: Spatial Distribution Function (SDF) — reference-frame 3-D density + orientation
status: draft
created: 2026-06-26
---

# Spatial Distribution Function (SDF) — reference-frame 3-D density + orientation

## Summary
Add the **Spatial Distribution Function** — the 3-D number density of a target
species in the *local body-fixed frame of a reference molecule*, accumulated over
a trajectory. Unlike the lab-frame `GaussianDensity` molrs already has, the SDF
first superimposes each frame's reference molecule onto a canonical orientation
(Kabsch alignment on a chosen 3+-atom reference set), applies the same rigid
transform to the surrounding target atoms, and only then bins them into a 3-D
grid. The result is the familiar "density cloud" showing *where* solvent/counter-
ions sit around a molecule. A companion **solvent-orientation** map records the
mean orientation of a target vector per voxel (TRAVIS's combined SDF + dipole
orientation).

This closes a distinct TRAVIS gap: molrs's density analyzers are lab-frame and
cannot answer "where, relative to this molecule, does the second species reside".

Library feature only — **no CLI**, no bindings, no isosurface rendering.

## Domain basis
- Spatial Distribution Functions: Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**,
  51, 2007–2023; Kusalik & Svishchev, *Science* **1994**, 265, 1219 (the original
  SDF for liquid water); Brehm et al., *J. Chem. Phys.* **2020**, 152, 164105.
- **Reference-frame transform**: for each frame, find the rigid rotation `R` and
  translation `t` minimizing RMSD of the reference atoms to a canonical template
  (Kabsch, *Acta Cryst.* **1976**, A32, 922). Apply `r' = R(r − r_com)` to target
  atoms (after minimum-image unwrapping relative to the reference COM), bin into a
  grid centered on the reference COM.
- **Normalization**: voxel counts → number density (Å⁻³), optionally divided by the
  bulk number density to give a dimensionless enhancement `g_SDF(x,y,z)` (the SDF
  analogue of RDF's `g(r)`).
- **Orientation map**: per voxel, accumulate the mean of a target unit vector
  (e.g. a solvent dipole direction) in the reference frame.

## Design
New submodule `compute/density/spatial.rs` (extends the existing `density` family,
`compute` feature).

New symbols:
- `kabsch(ref_template, ref_frame) -> (R: F3x3, rmsd)` — a small native Kabsch
  superposition on the 3×3 covariance, reusing molrs's existing 3×3 symmetric
  eigen/SVD path already used by `gyration_tensor`/`inertia_tensor` (no BLAS).
- `SpatialDistribution` — implements `Compute`; args: reference selection + canonical
  template, target selection, grid spec (extent + resolution). Per frame: align,
  transform targets (min-image unwrap vs. reference COM), accumulate the 3-D grid.
- `SpatialDistributionResult` — 3-D `ndarray` density (raw counts + Å⁻³ + optional
  bulk-normalized `g_SDF`) + grid metadata; plus an optional voxel orientation
  field (mean vector per voxel) when a target-vector observable is supplied.
- Reuses `GaussianDensity`'s grid + `wrap_index` smearing conventions for the
  binning so SDF and GaussianDensity share one grid implementation; SDF adds only
  the per-frame alignment front-end.

Layer discipline: `compute` → `core`; native Kabsch keeps it BLAS-free and
WASM-clean.

## Files to create or modify
- `molrs/src/compute/density/spatial.rs` (new) — `SpatialDistribution` + result.
- `molrs/src/compute/density/kabsch.rs` (new) — native Kabsch superposition (or reuse a core 3×3 eigen helper).
- `molrs/src/compute/density/mod.rs` (modify) — re-exports.
- `molrs/src/compute/mod.rs` (modify) — re-export `SpatialDistribution`.
- `molrs/tests/compute/spatial_distribution.rs` (new) — alignment + density + orientation tests.

## Tasks
- [ ] Write failing Kabsch tests: aligning a randomly rotated/translated rigid copy back to its template recovers R (det=+1, no reflection) with RMSD ≈ 0 within 1e-9.
- [ ] Write failing SDF test: a target placed at a fixed body-frame offset around a tumbling reference molecule accumulates a single sharp density voxel at that offset (lab-frame GaussianDensity would smear it into a shell).
- [ ] Write failing orientation test: a target vector fixed in the reference frame yields a uniform mean-vector field; a randomized one averages toward zero.
- [ ] Implement native Kabsch (covariance + 3×3 eigen, reflection guard) reusing the existing tensor eigen path.
- [ ] Implement `SpatialDistribution` (align → unwrap → bin) + bulk normalization + optional orientation field.
- [ ] Rustdoc with Kabsch/SDF equations + citations; run fmt/clippy/test.

## Testing strategy
- **Alignment** (`scientific`): exact recovery of a known rotation; proper-rotation
  (det +1) reflection guard verified on a chiral template.
- **Frame invariance**: the SDF of a rigidly co-moving reference+target is a single
  voxel regardless of global tumbling — the property that distinguishes SDF from
  lab-frame density.
- **Normalization**: ∑ voxels × voxel_volume × bulk_density consistency; `g_SDF → 1`
  far from the reference for an ideal-gas target.
- **PBC**: target unwrapping relative to the reference COM across a box edge.
- **Edge cases**: reference with <3 non-collinear atoms → typed error; empty target.

## Third-party library analysis
- **Kabsch / SVD**: compared (a) native 3×3 Kabsch on the covariance matrix reusing
  molrs's existing symmetric-eigensolver (used by gyration/inertia tensors),
  (b) `nalgebra` SVD, (c) `ndarray-linalg` SVD. **Recommend native (a)** — it is a
  3×3 problem, needs no BLAS, keeps the wasm32 target intact, and reuses code molrs
  already ships. `ndarray-linalg` is rejected (mandatory system BLAS, not WASM);
  `nalgebra` is rejected (large new dependency for one 3×3 SVD).
- **Grid/binning**: reuse the existing `GaussianDensity` grid; no new crate.
- *(Versions to confirm at implementation time.)*

## Out of scope
- Isosurface extraction / volumetric file export / rendering (visualization layer).
- Non-rigid / flexible reference alignment (RMSD-fit only on a rigid reference set).
- Cube/volumetric electron density — that is link 07.
- CLI, bindings.
