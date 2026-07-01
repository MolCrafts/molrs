---
title: Voronoi electron-density integration → molecular electromagnetic moments
status: draft
created: 2026-06-26
---

# Voronoi electron-density integration → molecular electromagnetic moments

## Summary
Add the capability that underpins TRAVIS's *ab-initio* spectroscopy: integrate a
volumetric **electron density** over each radical-Voronoi cell (link 06) to obtain
per-atom and per-molecule **electromagnetic moments** — total charge, dipole, and
(optionally) quadrupole — and, from finite-field density sets, molecular
**polarizabilities**. This is the missing geometric/physics bridge between an AIMD
trajectory (positions + a Gaussian-Cube electron density per frame) and the dipole/
polarizability time series that the spectra in link 08 consume.

Two new pieces are required:

1. **Cube trajectory IO** — molrs reads a *single* Gaussian Cube today
   (`io::data`); this adds a multi-frame **Cube trajectory** reader (a sequence of
   cube files / a concatenated cube stream) under `io::trajectory`.
2. **Voronoi integration** — assign each density grid point to its enclosing radical
   cell and accumulate per-cell electronic charge `q = −∫ρ dV` and electronic dipole
   `μ = −∫ρ (r − r_ref) dV`, combine with nuclear charges, and sum per molecule.

Library feature only — **no CLI**, no bindings.

## Domain basis
- Voronoi integration of electron density into molecular dipoles/polarizabilities:
  Thomas, Brehm, Kirchner, *Phys. Chem. Chem. Phys.* **2015**, 17, 3207 ("Voronoi
  dipole moments for the simulation of bulk phase vibrational spectra"); Brehm &
  Kirchner *J. Chem. Inf. Model.* **2011**, 51, 2007; Brehm et al. *J. Chem. Phys.*
  **2020**, 152, 164105.
- **Definitions** (electron density ρ ≥ 0): cell electronic charge
  `q_c^elec = −∫_{cell} ρ(r) dV`; molecular charge `Q_m = Σ_{a∈m} (Z_a − N_a^elec)`;
  molecular dipole `μ_m = Σ_a Z_a (r_a − r_ref) − ∫_{cells∈m} ρ(r)(r − r_ref) dV`,
  with a documented reference point (molecular COM or center of charge) and the
  origin-dependence noted for charged species.
- **Polarizability** `α_ij = ∂μ_i/∂E_j` via finite field: central differences over
  cube sets computed with small applied fields ±E_j (caller supplies the field-on
  density cubes; molrs does the differencing).
- **Units**: Cube files are atomic units (Bohr, e/Bohr³). Convert at the reader
  boundary to molrs units (Å, e) exactly as other readers normalize at their edge.

## Design
- **IO**: `io/trajectory/cube_traj.rs` — a `CubeTrajectory` reader yielding one
  `(grid, density, atoms)` per frame through the existing streaming/reader traits;
  Bohr→Å normalization at the boundary. Reuses the single-cube parser in
  `io::data` for the per-frame body.
- **Integration**: `compute/voronoi/integrate.rs` — `VoronoiIntegration`: given a
  built `RadicalVoronoi` (link 06) + a density grid, assign each voxel to its
  enclosing cell (point-in-cell via the cell's bounding half-spaces, with a
  fast nearest-radical-site shortcut) and accumulate `q`, `μ` (and optional
  quadrupole) per cell, then per molecule via a supplied atom→molecule map.
  `MolecularMoments` result: per-molecule charge + dipole (+ quadrupole), per frame.
- **Polarizability**: `polarizability_finite_field(moments_zero, moments_plus,
  moments_minus, field)` → per-molecule `α` tensor via central difference.
- Grid–cell assignment reuses link-06 cell geometry; no new tessellation.

Layer discipline: `io` for the reader; `compute/voronoi` (feature `voronoi`) for
integration; both `→ core`. The integrator is native + WASM-clean (the cube *data*
comes from an external AIMD code; molrs only reads + integrates).

## Files to create or modify
- `molrs/src/io/trajectory/cube_traj.rs` (new) — `CubeTrajectory` multi-frame reader.
- `molrs/src/io/trajectory/mod.rs` (modify) — register the reader.
- `molrs/src/compute/voronoi/integrate.rs` (new) — `VoronoiIntegration` + `MolecularMoments`.
- `molrs/src/compute/voronoi/polarizability.rs` (new) — finite-field `α`.
- `molrs/src/compute/voronoi/mod.rs` (modify) — re-exports.
- `molrs/tests/io/data/cube_traj.rs` (new) — trajectory reader tests (real files; see IO rule).
- `molrs/tests/compute/voronoi_integration.rs` (new) — integration + moment tests.
- `tests-data/cube_traj/` (new) — real multi-frame cube fixtures (added to the tests-data repo per the IO rule).

## Tasks
- [ ] Add real multi-frame cube fixtures to the tests-data repo under `cube_traj/`; write the failing trajectory-reader tests iterating every file (per the IO testing rule).
- [ ] Implement `CubeTrajectory` (multi-frame, Bohr→Å normalization) reusing the single-cube parser.
- [ ] Write failing integration tests: total integrated electronic charge over all cells equals the grid's total (∫ρ dV) within grid tolerance; a neutral atom's cell integrates to ≈ −Z electrons.
- [ ] Write failing dipole test: a constructed point/charge distribution on the grid integrates to the analytic dipole; a symmetric (e.g. centrosymmetric) molecule gives ≈ 0 dipole.
- [ ] Implement `VoronoiIntegration` (voxel→cell assignment + per-cell/per-molecule accumulation) and the finite-field polarizability.
- [ ] Rustdoc with the charge/dipole/α equations + unit conversion + citations; run fmt/clippy/test under `--features voronoi`.

## Testing strategy
- **Charge conservation** (`scientific`): `Σ_cell ∫ρ = ∫_grid ρ` within grid
  discretization tolerance; per-molecule total charge sums to the system charge.
- **Analytic dipole**: a known grid charge distribution integrates to its analytic
  dipole; centrosymmetric density → ~zero dipole; origin-dependence documented +
  tested for a charged species.
- **Units**: Bohr→Å and e/Bohr³→e conversions verified against a hand value.
- **IO**: trajectory reader iterates every real fixture file (never synthetic) and
  round-trips frame count/grid shape.
- **Polarizability**: a linear-response synthetic set recovers the input `α` tensor.
- **Edge cases**: a voxel exactly on a cell boundary assigned deterministically; a
  molecule split across the periodic boundary handled via min-image.

## Third-party library analysis
- **Cube parsing**: reuse molrs's existing single-Cube reader; no crate.
- **Voronoi geometry**: reuse link-06 `RadicalVoronoi` (native, WASM-clean); no crate.
- **Voxel→cell assignment**: nearest-radical-site test is a weighted nearest-neighbor
  query — reuse `core::NeighborQuery`; no crate. (A full point-in-polytope test is
  the fallback for boundary voxels.)
- No new third-party dependency is required; the heavy lifting (tessellation) was
  the link-06 build-vs-buy decision (native port chosen there for WASM).
- *(Versions to confirm at implementation time.)*

## Out of scope
- Computing the electron density itself (that is the upstream AIMD/DFT code; molrs
  reads the cube it produces).
- Maximally-localized Wannier-center dipoles (an alternative to Voronoi dipoles —
  note as a possible future link).
- The spectra built from these moments — **link 08**.
- CLI, bindings.
