---
title: Geometric distribution functions — ADF, DDF, distance DF + Observable extractors
status: draft
created: 2026-06-26
---

# Geometric distribution functions — ADF, DDF, distance DF + Observable extractors

## Summary
Add user-defined **geometric distribution functions** to `compute`: an angular
distribution function (ADF — the angle θ formed by selected atom triples or two
selected vectors), a dihedral distribution function (DDF — the torsion φ over
selected atom quadruples), and a general **distance** distribution function over
arbitrary selected atom pairs (the pairwise RDF is only the all-vs-all special
case). All three are built on one small reusable abstraction — an `Observable`
that maps a frame + an atom-group selection to a stream of scalar samples — and a
generic 1-D `DistributionFunction` that histograms those samples across a
trajectory, reusing the existing RDF/`density` histogram + `SimBox` minimum-image
conventions.

This is the foundation the **Combined Distribution Functions** (link 02) and the
reference-frame **Spatial Distribution Function** (link 03) build on, and it
closes the most basic TRAVIS-parity gap: molrs today exposes only the pairwise
`RDF` and a quaternion-based `environment::AngularSeparation` (a freud rigid-body
*orientation* metric), neither of which is the geometric ADF/DDF over
user-selected atoms that liquid-state and AIMD analysis depends on.

This is a library feature only — **no CLI**, no bindings, no plotting.

## Domain basis
- TRAVIS distribution functions: Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**,
  51, 2007–2023 (TRAVIS); Brehm, Thomas, Gehrke, Kirchner, *J. Chem. Phys.*
  **2020**, 152, 164105.
- **Angle** (three atoms i–j–k, j the vertex):
  `θ = arccos( (r_ij · r_kj) / (|r_ij| |r_kj|) )`, θ ∈ [0, π].
- **Dihedral** (four atoms i–j–k–l): the IUPAC-signed torsion via
  `φ = atan2( (b1 × b2)·(b2/|b2|) , (b1 × b2)·(b3 × b2) )` with
  `b1 = r_j−r_i, b2 = r_k−r_j, b3 = r_l−r_k`, φ ∈ (−π, π].
- **Normalization**: the result is a probability density (∫ p dx = 1) over the bin
  range; an ADF additionally offers a solid-angle (`sin θ`) correction — molrs
  emits **both** the raw and the `sin θ`-weighted ADF and documents the difference,
  because the two conventions are routinely confused.
- All interatomic vectors use the minimum-image convention under PBC, consistent
  with `compute::rdf`.

## Design
New feature-gated submodule `compute/distribution/` (under the existing `compute`
feature; no new dependency).

New symbols:
- `Observable` trait — `fn sample(&self, frame: &F, groups: &AtomGroups) -> Vec<f64>`:
  a stateless per-frame extractor turning each selected tuple into one scalar
  sample. Mirrors the existing stateless `Compute` contract (pure `&self`).
- `DistanceObservable`, `AngleObservable`, `DihedralObservable` — the three
  concretes, each consuming a 2/3/4-column `AtomGroups`.
- `AtomGroups` — a thin, frozen index-tuple container (`Vec<[u32; N]>` per arity)
  describing *which* atoms each sample is computed over. If a reusable selection
  type already exists in `core`, wrap it; otherwise this minimal container is the
  selection surface and later links extend it. (No new selection DSL here.)
- `DistributionFunction<O: Observable>` — implements `Compute`; accumulates a 1-D
  histogram (`bins`, `[min, max]`, optional `sin θ` weight flag) across all frames
  and returns `DistributionResult` (bin centers + edges, raw counts, normalized
  density). `finalize()` performs the ∫ = 1 normalization (and the sin-weight
  division for the corrected ADF), matching how `RDFResult::finalize` normalizes.

Reuse, do not duplicate: the bin/edge math and the `SimBox` minimum-image helpers
are lifted from `compute::rdf`/`compute::density` into a shared `histogram1d`
helper so RDF and the new DFs share one implementation.

Layer discipline: lives entirely inside `compute` (→ `core`); no `io`/`ff`/`signal`
coupling; WASM-clean (no BLAS, no FFI).

## Files to create or modify
- `molrs/src/compute/distribution/mod.rs` (new) — module + re-exports.
- `molrs/src/compute/distribution/observable.rs` (new) — `Observable` trait + `AtomGroups`.
- `molrs/src/compute/distribution/distance.rs` (new) — `DistanceObservable`.
- `molrs/src/compute/distribution/angle.rs` (new) — `AngleObservable` (+ vector-pair form).
- `molrs/src/compute/distribution/dihedral.rs` (new) — `DihedralObservable`.
- `molrs/src/compute/distribution/histogram1d.rs` (new) — shared 1-D histogram + normalization.
- `molrs/src/compute/mod.rs` (modify) — `pub mod distribution;` + re-exports.
- `molrs/tests/compute/distribution.rs` (new) — analytic + PBC + normalization tests.

## Tasks
- [ ] Write failing tests: equilateral-triangle frame → ADF density peaks at 60°; eclipsed/planar quadruple → DDF peak at 0°/180°; distance DF on a known lattice matches hand-computed separations.
- [ ] Write failing PBC test: an angle/distance spanning a periodic boundary uses the minimum image, matching `compute::rdf` behavior.
- [ ] Implement `Observable` + `AtomGroups` and the three concrete observables (angle via `arccos`, dihedral via `atan2`, distance via min-image).
- [ ] Implement `histogram1d` (extracted from rdf/density) and `DistributionFunction<O>` with raw + density + `sin θ`-corrected normalization.
- [ ] Add rustdoc with the cited θ/φ equations + the raw-vs-`sinθ` ADF convention note.
- [ ] Wire re-exports in `compute/mod.rs`; run `cargo fmt && clippy -D warnings && test --features compute`.

## Testing strategy
- **Analytic** (`scientific`): closed-form geometries (equilateral triangle, regular
  tetrahedron dihedrals, simple cubic lattice distances) with f64 tolerance 1e-9.
- **Normalization**: ∫ density dx = 1 within 1e-6 for each DF; raw counts sum to
  `n_samples × n_frames`.
- **PBC**: a pair/triple straddling the box edge yields the wrapped value.
- **Edge cases**: collinear triple → angle 0/π without NaN; zero-length vector →
  documented error (not silent NaN); empty `AtomGroups` → empty result, not panic.
- **Independent reference**: ADF of a small fixed configuration cross-checked
  against a NumPy snippet embedded in the test as expected constants.

## Third-party library analysis
- **Histogramming**: evaluated `ndhistogram` (clean axis API, MIT/Apache-2.0) vs.
  plain `ndarray`. **Recommend native `ndarray`** — molrs already hand-rolls the
  RDF/`density` histograms on `ndarray`; adding `ndhistogram` would fork the
  histogram convention and add a dependency for no WASM/perf benefit. The shared
  `histogram1d` helper keeps one implementation.
- **Geometry primitives**: angle/dihedral need only `acos`/`atan2`/`sqrt`, already
  available via the existing `libm` dependency; no crate required.
- *(Versions to confirm at implementation time.)*

## Out of scope
- Joint/combined histograms of 2–3 observables — **link 02 (CDF)**.
- 3-D reference-frame spatial density — **link 03 (SDF)**.
- A selection mini-language / SMARTS-driven groups (use explicit `AtomGroups`;
  SMARTS lives under the `smiles` feature and is not pulled in here).
- CLI, Python/WASM bindings, plotting.
