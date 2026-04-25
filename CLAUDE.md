# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

molrs is a Rust workspace for molecular simulation: core data structures, file I/O, trajectory analysis, force fields, 3D coordinate generation, molecular packing, and a CXX bridge to Atomiverse C++. Rust edition 2024, resolver "3".

## IO Testing Rules (MANDATORY)

**NEVER write synthetic/hand-crafted test data for IO tests.**

Every file-format reader/writer MUST be tested against **all** real files in
`molrs-core/target/tests-data/<format>/` (the test-data submodule is shared across
crates from this location). Rules:

1. When adding a new format reader (e.g. CHGCAR), add matching real files to
   the `tests-data` repo (`https://github.com/MolCrafts/tests-data`) under a
   new `<format>/` subdirectory before writing tests.
2. Tests iterate over **every** file in that directory — not a hardcoded subset.
   Use a helper that globs `tests-data/<format>/*` and runs assertions on each.
3. Unit tests inside `src/` may use `include_str!` with a **minimal** but
   structurally valid fixture only to cover parser edge-cases that are hard to
   produce from real data (e.g. malformed input → expected error). Keep these
   fixtures as small as possible and document where the snippet comes from.
4. Integration tests live in `molrs-io/tests/test_io/test_<format>.rs` and
   use `crate::test_data::get_test_data_path("<format>/<file>")`.

Violation: writing `let content = "..."; read_from_str(content)` for happy-path
format tests instead of reading a real file is **forbidden**.

## Build & Test Commands

```bash
# Build
cargo build
# Test (requires test data on first run)
bash scripts/fetch-test-data.sh      # clones to molrs-core/target/tests-data/
cargo test --all-features
cargo test -p molrs-core              # single crate
cargo test -p molrs-core test_name    # single test
cargo test --features slow-tests      # expensive integration tests
cargo test -p molrs-io                # IO format tests (uses tests-data submodule)

# Lint & Format
cargo fmt --all
cargo clippy -- -D warnings

# Benchmarks (criterion)
cargo bench -p molrs-core
```

## Workspace Crates & Dependency Flow

7 active workspace members. `molrs-core` is the foundation; everything else depends on it
(plus, in a few cases, on each other as listed):

```
molrs-core ── molrs-io ── molrs-cxxapi
            ── molrs-compute
            ── molrs-smiles
            ── molrs-ff (may also depend on molrs-io for parameter files)
            ── molrs-embed
```

| Crate | Purpose |
|---|---|
| `molrs-core` | Frame/Block/Grid/MolGraph/MolRec/Topology/Element, neighbors, math, region (SimBox), stereochemistry, rings, Gasteiger charges, hydrogen perception, atom-type mapping |
| `molrs-io` | File I/O: PDB, XYZ, LAMMPS data/dump, CHGCAR, Gaussian Cube, Zarr V3 trajectories |
| `molrs-compute` | Trajectory analysis: RDF, MSD, clustering, gyration/inertia tensors |
| `molrs-smiles` | SMILES parser → MolGraph |
| `molrs-ff` | Force fields, potentials (KernelRegistry), atom typifier |
| `molrs-embed` | 3D coordinate generation: distance geometry, fragment assembly, optimizer, rotor search |
| `molrs-cxxapi` | CXX bridge to Atomiverse C++ (zero-copy I/O via `FrameView`) |

Molecular packing (Packmol port) used to live here as `molrs-pack`; it now lives in the
standalone repo `MolCrafts/molpack` (crates.io: `molcrafts-molpack`, PyPI:
`molcrafts-molpack`). Add it as a separate dependency when needed.

Dirs `molrs-ffi/`, `molrs-wasm/`, `molrs-capi/`, `molrs-python/` exist on disk but are NOT
workspace members; treat as inactive / future work.

## Feature Flags

- `rayon` (default) — parallel neighbor lists and potentials
- `igraph` (default) — graph algorithms for molecular topology
- `zarr` / `filesystem` — Zarr V3 trajectory I/O
- `blas` — BLAS-backed linear algebra via `ndarray-linalg`
- `f64` — **deprecated / no-op** (F is now always f64)
- `slow-tests` — expensive integration tests

## Core Data Model

### Type Precision Principle

**Scientific algorithms use high precision; estimation / general code uses natural types.**

- `F = f64` always. The `f64` feature flag is deprecated and ignored.
- `I = i32` always. The `i64` feature flag is deprecated and ignored.
- `U = u32` always. The `u64` feature flag is deprecated and ignored.
- Neighbor list internals use `u32` directly — that's the natural type, no alias needed.
- The CXX bridge to Atomiverse still uses a `{F}` template that is resolved at build time by `cfg!(feature = "f64")` — this feature is set by CMake/corrosion to match Atomiverse's `ATV_REAL`.

Key type aliases: `F3 = Array1<F>`, `F3x3 = Array2<F>`, `FN = Array1<F>`, `FNx3 = Array2<F>`. Since `F = f64`, these are all double precision.

### Block (heterogeneous column store)

`Block` maps string keys to typed ndarray columns (f32, f64, i64, bool). Enforces consistent `nrows` across all columns. Type-safe access via `get_float()`, `get_int()`, `get_bool()`, `get_uint()`, `get_u8()`, `get_string()`. (`molrs-core/src/block/`).

### Frame (hierarchical data container)

`Frame` maps string keys (e.g. "atoms", "bonds", "angles") to `Block`s. Contains optional `SimBox` for periodic boundaries and a metadata hashmap. No forced cross-block row consistency — caller responsibility.

### MolGraph (molecular topology)

Graph-based molecular structure with atoms, bonds, stereochemistry, ring detection. Uses petgraph. (`molrs-core/src/molgraph.rs`).

## Trait-Based Extensibility

| Trait | Crate | Purpose | Key Implementations |
|---|---|---|---|
| `NbListAlgo` | `molrs-core::neighbors` | Neighbor search | `LinkCell` (O(N), default), `BruteForce` (O(N²), testing), `NeighborQuery` (high-level wrapper) |
| `Potential` | `molrs-ff::potential` | Energy/force evaluation | Bond harmonic, MMFF bond/angle/torsion/oop/vdw/ele, LJ/cut, PME |
| `Typifier` | `molrs-ff::typifier` | MolGraph → typed Frame | MMFFTypifier |

Pack-related traits (`Restraint`, `Region`, `Relaxer`, `Handler`, `Objective`) now live
in the standalone `molcrafts-molpack` crate.

## Key Subsystems

### Potential System (molrs-ff/src/potential/)

`KernelRegistry` maps `(category, style_name)` → `KernelConstructor`. Categories: bonds, angles, dihedrals, impropers, pairs, kspace. `ForceField::compile(frame)` resolves topology and constructs `Potentials` (aggregate sum). Coordinate format: flat `[x0,y0,z0, x1,y1,z1, ...]` (3N elements). MMFF94 parameters embedded at compile time from `data/mmff94.xml`.

### Free-Boundary Support

`SimBox::free(points, padding)` creates a non-periodic bounding box from atom positions. `NeighborQuery::free(points, cutoff)` auto-generates this box when no SimBox is present. RDF normalization (`molrs-compute`) falls back to bounding-box volume for free-boundary systems.

### Embed Pipeline (molrs-embed/)

Multi-stage 3D coordinate generation: distance geometry → fragment assembly → coarse minimization → rotor search → final minimization → stereo guards. Public API: `generate_3d(mol, opts) -> Result<(MolGraph, EmbedReport)>`.

### Packing

Lives in the standalone `MolCrafts/molpack` repository (crate
`molcrafts-molpack`, Python package `molcrafts-molpack` exposing `import
molpack`). Depends on `molcrafts-molrs-core` + `molcrafts-molrs-io`. See that
repo's docs for the Packmol-alignment workflow and associated conventions.

### FFI Layer (molrs-cxxapi/)

CXX bridge to Atomiverse C++. Zero-copy I/O via `FrameView` (borrowed) into existing `write_xyz_frame`; owned `Frame` only built when persisting to MolRec (Zarr). Bridge generated from `#[cxx::bridge]` in `bridge.rs`. No raw pointers cross the boundary.

## Critical Conventions

- **Coordinate format**: Potentials use flat `[x0,y0,z0, x1,y1,z1, ...]` vectors (3N elements), not Nx3 matrices.
- **`Cell<f64>` is NOT Sync**: Use `AtomicU64` with `f64::to_bits()`/`f64::from_bits()` for interior mutability in Sync contexts.

Packmol-port specific conventions (gradient sign, two-scale contract, LEFT
rotation multiplication) now live in the molpack repo's CLAUDE.md.

## Development Skills & Agents

Reference skills (the WHAT — standards) and agents (the HOW — executors) are
paired by domain. Reference skills are loaded by their paired agent and are
not user-invocable workflows.

| Domain | Reference skill | Agent |
|---|---|---|
| Architecture | `molrs-arch` | `molrs-architect` |
| Performance | `molrs-perf` | `molrs-optimizer` |
| Documentation (rustdoc) | `molrs-doc` | `molrs-documenter` |
| Testing | `molrs-test` | `molrs-tester` |
| Scientific correctness | `molrs-science` | `molrs-scientist` |
| FFI safety | `molrs-ffi` | `molrs-ffi-safety` |
| Documentation (docs system) | `molrs-docs` (workflow) | `molrs-docs-engineer` |

Workflow skills (user-invocable):

- `/molrs-impl <feature>` — orchestrator for new feature work; Plan → TDD → Implement → Verify → Document.
- `/molrs-spec <requirement>` — NL requirement → spec in `.claude/specs/` with an index entry.
- `/molrs-review [path]` — parallel multi-axis review; aggregates all agents above.
- `/molrs-fix <bug>` — minimal-diff bug fix, regression test first.
- `/molrs-refactor <scope>` — in-place restructure without behavior change; enforces hot-path extraction discipline.
- `/molrs-debug <symptom>` — read-only diagnosis; never edits.
- `/molrs-docs <kind>` — docs-system operations (Zensical, `.pyi`, READMEs, `docs.yml`).
- `/molrs-note <decision> | sweep | promote <slug>` — capture evolving decisions into `.claude/NOTES.md`; promote stable entries into this CLAUDE.md.

Evolving decisions live in `.claude/NOTES.md`; specs live in `.claude/specs/`
indexed by `.claude/specs/INDEX.md`. See `.claude/skills/` and
`.claude/agents/` for the full text of each.
