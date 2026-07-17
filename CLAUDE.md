---
mol_project:
  name: molrs
  language: rust
  stage: experimental
  build:
    install: "cargo build && bash scripts/fetch-test-data.sh"
    check: "cargo fmt --all --check && cargo clippy --all-targets --all-features -- -D warnings && cargo check --all-features && RUSTDOCFLAGS='-D warnings' cargo doc --no-deps -p molcrafts-molrs"
    test: "cargo test --all-features"
    test_single: "cargo test {path}"
  arch:
    style: crate-graph
    rules_section: "## Crate Structure & Modules"
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
`molrs/`). Sub-systems are modules under `molrs/src/`. **Three are always
compiled** — `core`, `perceive`, `optimize` — and the rest gate on a matching
feature. `core` and `perceive` are re-exported at the crate root (so
`molrs::Frame`, `molrs::system::…`, `molrs::find_rings`, `molrs::SmartsPattern`
resolve). The dependency spine is `core → perceive → {io, ff} → conformer`
(`compute` → `signal`, `conformer` → `ff`). In-crate paths use `crate::core::…` /
`crate::perceive::…` / `crate::io::…` (and `molrs::…` via
`extern crate self as molrs;`).

| Module (`molrs/src/`) | Feature | Purpose |
|---|---|---|
| `core` | always on | Frame/Block/Grid/MolGraph/MolRec/Topology/Element, neighbors, math, region (SimBox), graph hash, atom-type mapping |
| `perceive` | always on | **Chemical perception**, one layer above `core` and below `ff`/`io`/`conformer`: rings (SSSR), aromaticity, hydrogen perception, stereochemistry, rotatable bonds, Gasteiger charges, SMARTS/SMIRKS. Builder API `Perceive::new().find_*(&MolGraph) -> MolGraph` (graph-in/graph-out, non-mutating); the free functions it wraps are also re-exported at the crate root |
| `optimize` | always on | Numerical optimizers |
| `io` | `io` | File I/O: PDB, XYZ, LAMMPS data/dump, CHGCAR/POSCAR, Gaussian Cube, CIF, mol2, SDF, GRO, DCD, GROMACS TRR/XTC, Zarr V3 trajectories; SMILES parsing in `io/smiles/` (gated by `smiles`). **SMARTS lives in `perceive/smarts/`, not here** |
| `signal` | `signal` | Signal processing: FFT-based autocorrelation, window functions, frequency grids |
| `compute` | `compute` (→ `signal`) | Trajectory analysis: RDF, MSD, clustering, gyration/inertia tensors |
| `stream` | `stream` | Frame/Block serde + MessagePack/JSON `frame_to_bytes` |
| `ff` | `ff` | Force fields, potentials (KernelRegistry), atom typifier |
| `conformer` | `conformer` (→ `ff`) | 3D conformer generation: ETKDGv3 distance geometry, experimental-torsion refinement, MMFF94 cleanup, stereo guards |

The umbrella feature `full` enables every gated sub-system module; core knobs are
`rayon` (default), `zarr`, `filesystem`, `blas`, `voronoi`.

**Why `perceive` is always-on rather than feature-gated:** `core` is always
compiled and this code used to live *inside* it, so every consumer configuration
already compiled it. Keeping it unconditional reproduces the existing build graph
exactly — gating it would be a behaviour change, not a refactor. (Gating it later
to shrink the WASM bundle is a legitimate, separately-measured follow-up.)

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
  `conformer` (→ `ff`), `smiles` (→ `io`); `full` enables
  all of them. Each gates its module **and** its unique optional deps, so a build
  with a sub-system off does not compile that sub-system's dependency.
- Core knobs: `rayon` (default; parallel neighbor lists / potentials),
  `zarr` (Zarr V3 + `From<zarrs::*Error>` conversions), `filesystem`
  (→ `zarr`, filesystem store), `blas` (BLAS via `ndarray-linalg`),
  `slow-tests` (expensive integration tests).

## Core Data Model

### Type Precision Principle

**Scientific algorithms use high precision; estimation / general code uses natural types.**

- `F = f64`, `I = i32`, `U = u32` — always. There is no compile-time precision switch (the former `f64` / `i64` / `u64` Cargo features were removed).
- Neighbor list internals use `u32` directly — that's the natural type, no alias needed.
- The CXX bridge to Atomiverse crosses all coordinates / fields / distances as `f64` unconditionally (the merged crate hardcodes `F = f64`); there is no `cfg!(feature = "f64")` precision switch.

Key type aliases: `F3 = Array1<F>`, `F3x3 = Array2<F>`, `FN = Array1<F>`, `FNx3 = Array2<F>`. Since `F = f64`, these are all double precision.

### Block (heterogeneous column store)

`Block` maps string keys to typed ndarray columns (f32, f64, i64, bool). Enforces consistent `nrows` across all columns. Type-safe access via `get_float()`, `get_int()`, `get_bool()`, `get_uint()`, `get_u8()`, `get_string()`. (`molrs/src/core/store/block/`).

### Frame (hierarchical data container)

`Frame` maps string keys (e.g. "atoms", "bonds", "angles") to `Block`s. Contains optional `SimBox` for periodic boundaries and a metadata hashmap. No forced cross-block row consistency — caller responsibility.

### MolGraph (molecular topology)

Graph-based molecular structure with atoms, bonds, stereochemistry, ring detection. Built on generational arenas (`slotmap`) with kind-tagged, multi-arity relations over a `smallvec`-backed adjacency map. (`molrs/src/core/system/molgraph.rs`). The connectivity graph is `Topology` (`molrs/src/core/system/topology.rs`), a native adjacency structure (`HashMap`/`VecDeque`, no petgraph) used for connectivity queries (connected components, BFS distances, angle/dihedral enumeration). SMARTS lives in `molrs/src/core/chem/smarts/` (moving to `molrs/src/perceive/smarts/`), and its subgraph-isomorphism matcher is hand-rolled (a backtracking VF2-style port of RDKit's `SubstructMatch.cpp`) — molrs has **no petgraph dependency at all**.

## Trait-Based Extensibility

| Trait | Crate | Purpose | Key Implementations |
|---|---|---|---|
| `NbListAlgo` | `molrs::core::neighbors` | Neighbor search | `LinkCell` (O(N), default), `BruteForce` (O(N²), testing), `NeighborQuery` (high-level wrapper) |
| `Potential` | `molrs::ff::potential` | Energy/force evaluation | Bond harmonic, MMFF bond/angle/torsion/oop/vdw/ele, LJ/cut, PME |
| `Typifier` | `molrs::ff::typifier` | MolGraph → typed Frame | `MMFF94Typifier` / `MMFF94STypifier` (one engine, two named front doors — the MMFF variant is a private field, never a constructor flag), `OPLSAATypifier`, `AtdTypifier` |

Pack-related traits (`Restraint`, `Region`, `Relaxer`, `Handler`, `Objective`) now live
in the standalone `molcrafts-molpack` crate.

## Key Subsystems

### Potential System (molrs/src/ff/potential/)

`KernelRegistry` maps `(category, style_name)` → `KernelConstructor`. Categories: bonds, angles, dihedrals, impropers, pairs, kspace. `ForceField::to_potentials(frame)` (with `Style::to_potential`) resolves topology and constructs `Potentials` (aggregate sum) — frame-free, deferred potentials that bind topology and coordinates at evaluation time. Coordinate format: flat `[x0,y0,z0, x1,y1,z1, ...]` (3N elements).

**Every parameter table is committed Rust, in one place.** `molrs/src/ff/params/` holds them all, flat — GAFF/GAFF2, the seven `ATOMTYPE_*.DEF` sets, BCCPARM, GASPARM, MMFF, OPLS-AA. Nothing parses parameter text at runtime, so a malformed table is a **compile** error, not a runtime one. `molrs/data/` and `molrs::data::*_XML` no longer exist. How a table *arrived* lives in its header doc (`scripts/gen_param_tables.py` from AmberTools' `.DAT`/`.DEF`; RDKit's `Params.cpp` for MMFF) — never in its name.

**A registered kernel constructor that ignores `tp` is not a Style.** `ParamSource::{TypeRows, PerInstance}` (`ff/potential/registry.rs`) makes that a type, and a bidirectional gate makes it a test. MMFF's parameters depend on aromaticity, ring size and equivalence degradation — no type-tuple table expresses them — so the typifier resolves them per instance and bakes them onto the Frame, and its kernels read those columns. That is legitimate; *registering as if it used type rows* was not.

**MMFF owns no kernel.** Its electrostatics is a buffered Coulomb — `pair/coul/cut` parameterised by `MMFF_ELE_STYLE` (`E = k·qᵢqⱼ / (D·(r+δ))`, δ = 0.05 Å; δ = 0 degenerates to the textbook form). A force field must **declare** the constants it means: a style that omits `coulomb` / `dielectric` / `coulomb14scale` is an `Err`, never a silent default. A kernel that supplies the force field's own constants is not reading the force field.

### Free-Boundary Support

`SimBox::free(points, padding)` creates a non-periodic bounding box from atom positions. `NeighborQuery::free(points, cutoff)` auto-generates this box when no SimBox is present. RDF normalization (`molrs::compute`) falls back to bounding-box volume for free-boundary systems.

### Conformer Pipeline (molrs/src/conformer/)

Multi-stage 3D coordinate generation: ETKDGv3 constraints → 4D distance-geometry embedding → experimental-torsion refinement → MMFF94 cleanup → stereo guards. Public API: `Conformer::new(opts).generate(mol) -> Result<(Atomistic, ConformerReport)>`.

### Packing

Lives in the standalone `MolCrafts/molpack` repository (crate
`molcrafts-molpack`, Python package `molcrafts-molpack` exposing `import
molpack`). Depends on `molcrafts-molrs` (with the `io` feature). See that
repo's docs for the Packmol-alignment workflow and associated conventions.

### FFI Layer (molrs-cxxapi/)

CXX bridge to Atomiverse C++. Zero-copy I/O via `FrameView` (borrowed) into existing `write_xyz_frame`; owned `Frame` only built when persisting to MolRec (Zarr). Bridge generated from `#[cxx::bridge]` in `bridge.rs`. No raw pointers cross the boundary.

### Consuming molrs from other projects

See `docs/interop.md` for the two as-built paths — **native Rust** (depend on `molcrafts-molrs`, use `molrs::Frame` / `molrs::ff::ForceField` directly; what molpack does) and **Python/WASM** (the `molrs-ffi` handle API: `FrameRef` / `BlockRef` / `ForceFieldRef` / `SharedStore`) — plus the shared data contract: **uint** atom indices, the `atomi/atomj/is_14` pairs-block schema, `special_bonds` weights on `ForceField`, and the consumer-built `intramolecular_pairs` neighbour list.

## Critical Conventions

- **Coordinate format**: Potentials use flat `[x0,y0,z0, x1,y1,z1, ...]` vectors (3N elements), not Nx3 matrices.
- **Angles are radians internally**: every angle-valued force-field parameter — angle `theta0`, dihedral/improper phase `d`/`phi`/`chi0` — is stored and consumed in **radians**. Kernel constructors do **no** angle-unit conversion; each *reader* normalizes user-facing degree inputs to radians at its boundary (`.to_radians()`), exactly as length/energy are normalized there (LAMMPS `*.ff` deg→rad, MMFF94 XML deg→rad; the OPLS/GROMACS reader is already radians). The molrs-native `<ForceField>` XML is an internal serialization and is therefore already radians.
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
