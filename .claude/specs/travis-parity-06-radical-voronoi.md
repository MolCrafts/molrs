---
title: Radical (Laguerre) Voronoi tessellation + domain & void analysis
status: draft
created: 2026-06-26
---

# Radical (Laguerre) Voronoi tessellation + domain & void analysis

## Summary
Add a **radical (Laguerre / power) Voronoi tessellation** core to molrs — 3-D,
periodic, weighted by per-atom radii — plus its first two non-electronic consumers,
**domain (microheterogeneity) analysis** and **void analysis**. The radical
tessellation is the geometric engine behind TRAVIS's Voronoi-based analyses and is
the prerequisite for the electron-density integration in link 07. Per cell it
yields volume, the bounding faces (area + the neighbor across each face), and the
neighbor list. Because molrs forbids shipping a subsystem with no production caller,
this link lands the tessellation **together with** two real consumers:

- **Domain analysis** — partition atoms into user-defined groups (e.g. polar vs.
  apolar), merge face-adjacent same-group cells, and report the resulting domain
  size distribution / continuity (ionic-liquid microheterogeneity).
- **Void analysis** — aggregate the unoccupied volume (large-radius probe cells /
  empty regions) into a cavity size distribution and total free volume.

Library feature only — **no CLI**, no bindings.

## Domain basis
- Radical Voronoi / power diagram: F. Aurenhammer, *SIAM J. Comput.* **1987**, 16,
  78; C. H. Rycroft, *Chaos* **2009**, 19, 041111 (voro++, the cell-by-cell radical
  Voronoi algorithm). The radical plane between atoms i, j with radii Rᵢ, Rⱼ sits at
  `|x−xᵢ|² − Rᵢ² = |x−xⱼ|² − Rⱼ²` (reduces to the plain Voronoi bisector when
  Rᵢ = Rⱼ).
- TRAVIS Voronoi/domain/void analysis: Brehm et al., *J. Chem. Phys.* **2020**, 152,
  164105; Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**, 51, 2007.
- Invariant: `Σ_cell volume = box volume` (periodic) — the headline numerical check.

## Design
New feature `voronoi` (gates the module + any optional backend dep; off by default,
included in `full`). New submodule `compute/voronoi/`.

New symbols:
- `RadicalVoronoi` — builds the periodic radical tessellation from atom positions +
  radii + `SimBox`. `VoronoiCells` result: per-cell `volume`, `faces` (each with
  `area` + `neighbor` index + optional face centroid/normal), and the neighbor list.
- `DomainAnalysis` — args: a per-atom group label + a built `VoronoiCells`; merges
  face-adjacent same-label cells (union-find over the cell-neighbor graph) and
  returns the domain size distribution + count + largest-domain fraction.
- `VoidAnalysis` — aggregates empty/low-occupancy volume into a cavity size
  distribution + total void fraction from the same tessellation.

Backend (see analysis): a **native pure-Rust** radical-Voronoi cell construction
(per-particle half-space clipping of an initially box-sized cell against radical
planes of neighbors, voro++ algorithm) — keeps the wasm32 target and avoids a C++
FFI. An **optional** `voro_rs` FFI backend MAY be added behind a separate
non-default, non-WASM feature purely as a cross-check oracle in tests; the shipped
default path is native.

Layer discipline: `compute` → `core` (`SimBox`, `NeighborQuery` for candidate
neighbors); WASM-clean on the default native backend.

## Files to create or modify
- `molrs/src/compute/voronoi/mod.rs` (new) — re-exports.
- `molrs/src/compute/voronoi/radical.rs` (new) — `RadicalVoronoi` + cell clipping.
- `molrs/src/compute/voronoi/cell.rs` (new) — `VoronoiCells` / per-cell geometry.
- `molrs/src/compute/voronoi/domain.rs` (new) — `DomainAnalysis` (union-find).
- `molrs/src/compute/voronoi/void.rs` (new) — `VoidAnalysis`.
- `molrs/src/compute/mod.rs` (modify) — gated re-exports.
- `molrs/Cargo.toml` (modify) — add `voronoi` feature (+ optional `voro_rs` behind a separate feature).
- `molrs/tests/compute/voronoi.rs` (new) — volume-sum + analytic-cell + domain/void tests.

## Tasks
- [ ] Write failing tessellation tests: total cell volume = box volume (1e-9); a simple cubic lattice (equal radii) gives unit-cube cells; a 2-atom unequal-radius case puts the face at the analytic radical plane.
- [ ] Write failing periodic test: cells wrap correctly across box edges; neighbor relation is symmetric (i lists j ⇔ j lists i).
- [ ] Write failing domain test: a constructed bilayer of two labels yields the expected two domains; an interpenetrating mix yields one percolating domain.
- [ ] Write failing void test: a lattice with a removed atom yields one cavity of ~the expected volume.
- [ ] Implement native periodic radical-Voronoi cell clipping (voro++ algorithm) reusing `NeighborQuery` for candidate neighbors.
- [ ] Implement `DomainAnalysis` (union-find over cell adjacency) + `VoidAnalysis`.
- [ ] Add the `voronoi` feature; rustdoc with radical-plane equation + citations; run fmt/clippy/test under `--features voronoi`.

## Testing strategy
- **Volume conservation** (`scientific`): `Σ volume = box volume` to 1e-9 — the
  single most important correctness check.
- **Analytic cells**: equal-radius cubic lattice → cubes; two-atom radical plane at
  the closed-form position; equal radii reproduce plain Voronoi.
- **Periodic + symmetry**: edge-wrapping; symmetric neighbor/face relations; face
  areas of adjacent cells agree.
- **Domain/void**: constructed labelings/cavities match hand-computed sizes.
- **Edge cases**: single atom in a periodic box → one cell = box; zero radius =
  plain Voronoi; degenerate cocircular points handled deterministically.
- *(Optional)* if the `voro_rs` oracle feature is built, cross-check native cell
  volumes against voro++ on a random configuration.

## Third-party library analysis
Goal: 3-D, **radical/weighted**, **periodic**, per-cell volume+faces+neighbors,
runnable on **wasm32** (molrs ships native + PyO3 + WASM).
- **`voro_rs`** — FFI bindings to voro++ (Rycroft, C++). Full radical + periodic +
  per-cell data — exactly the feature set. **But**: C++ FFI ⇒ **breaks wasm32**, adds
  a C++ build-toolchain requirement. voro++ license: BSD-3-Clause-style (LBNL) —
  redistribution-compatible. Use only as an optional native-only test oracle.
- **`voronator`** — 2-D only (delaunator port). ✗ no 3-D, ✗ no radical.
- **`spade`** — 2-D Delaunay/Voronoi. ✗ no 3-D.
- **`qhull` / `qhull-sys`** — N-D via Qhull (C FFI). Voronoi but **not radical/power**
  out of the box; C FFI ⇒ breaks wasm32; Qhull's license is a custom non-OSI license
  (redistribution caveats). ✗
- **Pure-Rust 3-D radical Voronoi crate**: none mature as of research.
- **Recommendation**: **native pure-Rust port of voro++'s per-cell radical-Voronoi
  clipping** (each cell starts as the box and is clipped by the radical half-space of
  each candidate neighbor from `NeighborQuery`). It is WASM-clean, dependency-free,
  BSD-compatible (a reimplementation citing the voro++ algorithm), and matches
  molrs's "one Rust implementation that runs everywhere" philosophy. Keep `voro_rs`
  as an **optional, non-default, non-WASM** test-oracle backend only.
- *(Versions/licenses to re-verify at implementation time.)*

## Out of scope
- Electron-density integration over cells → molecular EM moments — **link 07**.
- Triclinic (non-orthorhombic) cells in the first cut (orthorhombic first; note
  triclinic as a follow-up if `SimBox` triclinic support is needed).
- Solvent-accessible-surface / Connolly surfaces (different geometry).
- CLI, bindings.
