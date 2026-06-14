---
mol_project:
  name: molrs
  language: rust
  stage: experimental
  build:
    install: "cargo build && bash scripts/fetch-test-data.sh"
    check: "cargo fmt --all --check && cargo clippy -- -D warnings && cargo check"
    test: "cargo test --all-features"
    test_single: "cargo test {path}"
  arch:
    style: crate-graph
    rules_section: "## Workspace Crates & Dependency Flow"
  doc:
    style: rustdoc
  science:
    required: true
  notes_path: .claude/notes/notes.md
  specs_path: .claude/specs/
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

molrs is a Rust workspace for molecular simulation: core data structures, file I/O, trajectory analysis, signal processing, force fields, 3D coordinate generation, and a CXX bridge to Atomiverse C++. Rust edition 2024, resolver "3".

## IO Testing Rules (MANDATORY)

**NEVER write synthetic/hand-crafted test data for IO tests.**

Every file-format reader/writer MUST be tested against **all** real files in
`tests-data/<format>/` — a binding-neutral directory at the **workspace root**
(gitignored; cloned by `scripts/fetch-test-data.sh`), shared by every Rust crate
and by the Python / C / WASM bindings. Rules:

1. When adding a new format reader (e.g. CHGCAR), add matching real files to
   the `tests-data` repo (`https://github.com/MolCrafts/tests-data`) under a
   new `<format>/` subdirectory before writing tests.
2. Tests iterate over **every** file in that directory — not a hardcoded subset.
   Use the small local `common` helper in the io test target
   (`common::format_files("<format>")`) and run assertions on each.
3. **Inline `#[cfg(test)]` tests in `src/` are pure function unit tests only** —
   logic, edge cases, error paths. They must NOT read real files from
   `tests-data/`. A minimal `include_str!` fixture is permitted ONLY to cover a
   parser edge-case hard to produce from real data (e.g. malformed input →
   expected error); keep it tiny and document its origin.
4. Data-driven integration tests live in the merged crate's `tests/` tree,
   mirroring the `src/` module layout (e.g. `molrs/tests/io/data/<format>.rs`), and
   resolve files via the io test target's local `common` module
   (`common::{tests_data_dir, data_path, format_files}`), which simply reads
   `../tests-data` (or `$MOLRS_TESTS_DATA`). No helper crate.

Violation: writing `let content = "..."; read_from_str(content)` for happy-path
format tests instead of reading a real file is **forbidden**.

## Build & Test Commands

```bash
# Build
cargo build
# Test (requires test data on first run)
bash scripts/fetch-test-data.sh      # clones to <root>/tests-data/ (binding-neutral)
cargo test --all-features
cargo test -p molcrafts-molrs --features full   # all sub-system modules
cargo test test_name                            # single test
cargo test --features slow-tests                # expensive integration tests
cargo test --test io --features full            # IO format tests (iterate every file in tests-data/)

# Lint & Format
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Benchmarks (criterion)
cargo bench -p molcrafts-molrs --bench core_benchmarks
```

## Crate Structure & Modules

molrs is a **single published crate** `molcrafts-molrs` (lib name `molrs`, dir
`molrs/`). Its sub-systems are **feature-gated modules** under `molrs/src/`:
`core` is always compiled and re-exported at the crate root (so `molrs::Frame`,
`molrs::system::…` resolve); the rest gate on a matching feature. `core` is the
foundation; everything else depends on it (`compute` → `signal`, `conformer` →
`ff`). In-crate paths use `crate::core::…` / `crate::io::…` (and `molrs::…` via
`extern crate self as molrs;`).

| Module (`molrs/src/`) | Feature | Purpose |
|---|---|---|
| `core` | always on | Frame/Block/Grid/MolGraph/MolRec/Topology/Element, neighbors, math, region (SimBox), stereochemistry, rings, Gasteiger charges, hydrogen perception, atom-type mapping |
| `io` | `io` | File I/O: PDB, XYZ, LAMMPS data/dump, CHGCAR/POSCAR, Gaussian Cube, CIF, mol2, SDF, GRO, DCD, Zarr V3 trajectories; SMILES/SMARTS parsing in `io/smiles/` (gated by the `smiles` feature) |
| `signal` | `signal` | Signal processing: FFT-based autocorrelation, window functions, frequency grids |
| `compute` | `compute` (→ `signal`) | Trajectory analysis: RDF, MSD, clustering, gyration/inertia tensors |
| `ff` | `ff` | Force fields, potentials (KernelRegistry), atom typifier |
| `conformer` | `conformer` (→ `ff`) | 3D conformer generation: distance geometry, fragment assembly, optimizer, rotor search |

The umbrella feature `full` enables every sub-system module; core knobs are
`rayon` (default), `zarr`, `filesystem`, `blas`.

The other workspace member is `molrs-cxxapi` (`molcrafts-molrs-cxxapi`, a
`staticlib` CXX bridge to Atomiverse C++ via `FrameView`); it depends on the
merged crate with `features = ["io"]`. The binder crates `molrs-ffi`,
`molrs-wasm`, `molrs-python`, and `molrs-capi` are **separate workspaces** (not
members) and each depend on `molcrafts-molrs` with the features they need.

Molecular packing (Packmol port) used to live here as `molrs-pack`; it now lives in the
standalone repo `MolCrafts/molpack` (crates.io: `molcrafts-molpack`, PyPI:
`molcrafts-molpack`). Add it as a separate dependency when needed.

## Feature Flags

All on the single `molcrafts-molrs` crate (`molrs/Cargo.toml`):

- Sub-system modules: `io`, `signal`, `compute` (→ `signal`), `ff`,
  `conformer` (→ `ff`), `smiles` (→ `io`, pulls in `petgraph`); `full` enables
  all of them. Each gates its module **and** its unique optional deps, so a build
  with a sub-system off does not compile that sub-system's dependency.
- Core knobs: `rayon` (default; parallel neighbor lists / potentials),
  `zarr` (Zarr V3 + `From<zarrs::*Error>` conversions), `filesystem`
  (→ `zarr`, filesystem store), `blas` (BLAS via `ndarray-linalg`),
  `slow-tests` (expensive integration tests).
- `f64` / `i64` / `u64` — **deprecated / no-op** (F/I/U are hardcoded to
  f64/i32/u32); kept so binders that forward them keep resolving.

## Core Data Model

### Type Precision Principle

**Scientific algorithms use high precision; estimation / general code uses natural types.**

- `F = f64` always. The `f64` feature flag is deprecated and ignored.
- `I = i32` always. The `i64` feature flag is deprecated and ignored.
- `U = u32` always. The `u64` feature flag is deprecated and ignored.
- Neighbor list internals use `u32` directly — that's the natural type, no alias needed.
- The CXX bridge to Atomiverse crosses all coordinates / fields / distances as `f64` unconditionally (the merged crate hardcodes `F = f64`); there is no `cfg!(feature = "f64")` precision switch.

Key type aliases: `F3 = Array1<F>`, `F3x3 = Array2<F>`, `FN = Array1<F>`, `FNx3 = Array2<F>`. Since `F = f64`, these are all double precision.

### Block (heterogeneous column store)

`Block` maps string keys to typed ndarray columns (f32, f64, i64, bool). Enforces consistent `nrows` across all columns. Type-safe access via `get_float()`, `get_int()`, `get_bool()`, `get_uint()`, `get_u8()`, `get_string()`. (`molrs/src/core/store/block/`).

### Frame (hierarchical data container)

`Frame` maps string keys (e.g. "atoms", "bonds", "angles") to `Block`s. Contains optional `SimBox` for periodic boundaries and a metadata hashmap. No forced cross-block row consistency — caller responsibility.

### MolGraph (molecular topology)

Graph-based molecular structure with atoms, bonds, stereochemistry, ring detection. Built on generational arenas (`slotmap`) with kind-tagged, multi-arity relations over a `smallvec`-backed adjacency map. (`molrs/src/core/system/molgraph.rs`). The connectivity graph is `Topology` (`molrs/src/core/system/topology.rs`), a native adjacency structure (`HashMap`/`VecDeque`, no petgraph) used for connectivity queries (connected components, BFS distances, angle/dihedral enumeration). petgraph is pulled in only by the SMARTS VF2 matcher under the `smiles` feature (`molrs/src/io/smiles/`).

## Trait-Based Extensibility

| Trait | Crate | Purpose | Key Implementations |
|---|---|---|---|
| `NbListAlgo` | `molrs::core::neighbors` | Neighbor search | `LinkCell` (O(N), default), `BruteForce` (O(N²), testing), `NeighborQuery` (high-level wrapper) |
| `Potential` | `molrs::ff::potential` | Energy/force evaluation | Bond harmonic, MMFF bond/angle/torsion/oop/vdw/ele, LJ/cut, PME |
| `Typifier` | `molrs::ff::typifier` | MolGraph → typed Frame | MMFFTypifier |

Pack-related traits (`Restraint`, `Region`, `Relaxer`, `Handler`, `Objective`) now live
in the standalone `molcrafts-molpack` crate.

## Key Subsystems

### Potential System (molrs/src/ff/potential/)

`KernelRegistry` maps `(category, style_name)` → `KernelConstructor`. Categories: bonds, angles, dihedrals, impropers, pairs, kspace. `ForceField::to_potentials(frame)` (with `Style::to_potential`) resolves topology and constructs `Potentials` (aggregate sum) — frame-free, deferred potentials that bind topology and coordinates at evaluation time. Coordinate format: flat `[x0,y0,z0, x1,y1,z1, ...]` (3N elements). MMFF94/MMFF94s parameters are embedded at compile time in core (`molrs/data/mmff94.xml`, exposed as `molrs::data::MMFF94_XML`).

### Free-Boundary Support

`SimBox::free(points, padding)` creates a non-periodic bounding box from atom positions. `NeighborQuery::free(points, cutoff)` auto-generates this box when no SimBox is present. RDF normalization (`molrs::compute`) falls back to bounding-box volume for free-boundary systems.

### Conformer Pipeline (molrs/src/conformer/)

Multi-stage 3D coordinate generation: distance geometry → fragment assembly → coarse minimization → rotor search → final minimization → stereo guards. Public API: `Conformer::new(opts).generate(mol) -> Result<(Atomistic, ConformerReport)>`.

### Packing

Lives in the standalone `MolCrafts/molpack` repository (crate
`molcrafts-molpack`, Python package `molcrafts-molpack` exposing `import
molpack`). Depends on `molcrafts-molrs` (with the `io` feature). See that
repo's docs for the Packmol-alignment workflow and associated conventions.

### FFI Layer (molrs-cxxapi/)

CXX bridge to Atomiverse C++. Zero-copy I/O via `FrameView` (borrowed) into existing `write_xyz_frame`; owned `Frame` only built when persisting to MolRec (Zarr). Bridge generated from `#[cxx::bridge]` in `bridge.rs`. No raw pointers cross the boundary.

## Critical Conventions

- **Coordinate format**: Potentials use flat `[x0,y0,z0, x1,y1,z1, ...]` vectors (3N elements), not Nx3 matrices.
- **`Cell<f64>` is NOT Sync**: Use `AtomicU64` with `f64::to_bits()`/`f64::from_bits()` for interior mutability in Sync contexts.

Packmol-port specific conventions (gradient sign, two-scale contract, LEFT
rotation multiplication) now live in the molpack repo's CLAUDE.md.

## Development Workflow (mol plugin)

Process workflows come from the molcrafts `mol` plugin; project-specific
standards live in `.claude/notes/` topic pages (see next section).

Workflow skills (user-invocable):

- `/mol:spec <requirement>` — NL requirement → spec + acceptance contract in `.claude/specs/`.
- `/mol:impl <spec>` — orchestrator for new feature work; spec → TDD → implement → verify → simplify → close.
- `/mol:review [path]` — parallel multi-axis review (architecture, performance, science, FFI via `--axis=ffi`, …).
- `/mol:fix <bug>` — minimal-diff bug fix, regression test first.
- `/mol:refactor <scope>` — in-place restructure without behavior change; enforces hot-path extraction discipline (see `.claude/notes/performance.md`).
- `/mol:debug <symptom>` — read-only diagnosis; never edits.
- `/mol:docs <kind>` — docs work (rustdoc, Zensical site, `.pyi`, READMEs, `docs.yml`); site rules in `.claude/notes/docs.md`.
- `/mol:note <decision> | sweep | promote <slug>` — capture evolving decisions into `.claude/notes/notes.md`; promote stable entries into this CLAUDE.md.

Agent mapping by domain (plugin agents read the notes pages below):

| Domain | Standard (notes page) | Agent |
|---|---|---|
| Architecture | `.claude/notes/architecture-rules.md` | `mol:architect` |
| Performance | `.claude/notes/performance.md` | `mol:optimizer` |
| Documentation (rustdoc) | `.claude/notes/docs.md` (Part A) | `mol:documenter` |
| Documentation (docs system) | `.claude/notes/docs.md` (Part B) | `mol:documenter` |
| Testing | `.claude/notes/testing.md` | `mol:tester` |
| Scientific correctness | `.claude/notes/science.md` | `mol:scientist` |
| FFI safety | `.claude/notes/ffi.md` | `mol:ffi-guard` (`/mol:review --axis=ffi`) |

Evolving decisions live in `.claude/notes/notes.md`; specs live in
`.claude/specs/` indexed by `.claude/specs/INDEX.md`.
