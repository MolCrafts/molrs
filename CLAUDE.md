# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

molrs is a Rust workspace for molecular simulation: core data structures, molecular packing, FFI layer, and WebAssembly bindings. Rust edition 2024, resolver "3".

## IO Testing Rules (MANDATORY)

**NEVER write synthetic/hand-crafted test data for IO tests.**

Every file-format reader/writer MUST be tested against **all** real files in
`molrs-core/target/tests-data/<format>/`. Rules:

1. When adding a new format reader (e.g. CHGCAR), add matching real files to
   the `tests-data` repo (`https://github.com/MolCrafts/tests-data`) under a
   new `<format>/` subdirectory before writing tests.
2. Tests iterate over **every** file in that directory — not a hardcoded subset.
   Use a helper that globs `tests-data/<format>/*` and runs assertions on each.
3. Unit tests inside `src/` may use `include_str!` with a **minimal** but
   structurally valid fixture only to cover parser edge-cases that are hard to
   produce from real data (e.g. malformed input → expected error). Keep these
   fixtures as small as possible and document where the snippet comes from.
4. Integration tests live in `molrs-core/tests/test_io/test_<format>.rs` and
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

# Lint & Format
cargo fmt --all
cargo clippy -- -D warnings

# WASM
cd molrs-wasm && wasm-pack build --release --target web

# Benchmarks (criterion)
cargo bench -p molrs-core
```

## Workspace Crates & Dependency Flow

```
molrs-core ← molrs-ffi ← molrs-wasm
molrs-core ← molrs-pack
```

| Crate | Purpose |
|---|---|
| `molrs-core` | Core: Frame/Block/MolGraph, I/O (PDB, XYZ, LAMMPS, Zarr), neighbors, potentials, typifiers, gen3d |
| `molrs-pack` | Molecular packing: faithful Packmol port, GENCAN optimizer, geometric constraints |
| `molrs-ffi` | Handle-based FFI via SlotMap with version-tracked invalidation |
| `molrs-wasm` | wasm-bindgen browser bindings |

## Feature Flags

- `rayon` (default) — parallel neighbor lists and potentials
- `igraph` (default) — graph algorithms for molecular topology
- `zarr` / `filesystem` — Zarr V3 trajectory I/O
- `blas` — BLAS-backed linear algebra via `ndarray-linalg`
- `f64` — double-precision floats (default: f32)
- `slow-tests` — expensive integration tests

## Core Data Model

### Float Precision

`type F = f32` (or `f64` with feature flag). All numeric code uses the `F` alias from `molrs-core/src/types.rs`. Key type aliases: `F3 = Array1<F>`, `F3x3 = Array2<F>`, `FN = Array1<F>`, `FNx3 = Array2<F>`.

### Block (heterogeneous column store)

`Block` maps string keys to typed ndarray columns (f32, f64, i64, bool). Enforces consistent `nrows` across all columns. Type-safe access via `get_float()`, `get_int()`, `get_bool()`, `get_uint()`, `get_u8()`, `get_string()`. (`molrs-core/src/block/`).

### Frame (hierarchical data container)

`Frame` maps string keys (e.g. "atoms", "bonds", "angles") to `Block`s. Contains optional `SimBox` for periodic boundaries and a metadata hashmap. No forced cross-block row consistency — caller responsibility.

### MolGraph (molecular topology)

Graph-based molecular structure with atoms, bonds, stereochemistry, ring detection. Uses petgraph. (`molrs-core/src/molgraph.rs`).

## Trait-Based Extensibility

| Trait | Module | Purpose | Key Implementations |
|---|---|---|---|
| `NbListAlgo` | `neighbors/` | Neighbor search | `LinkCell` (O(N), default), `BruteForce` (O(N²), testing), `NeighborQuery` (high-level wrapper) |
| `Potential` | `potential/` | Energy/force evaluation | Bond harmonic, MMFF bond/angle/torsion/oop/vdw/ele, LJ/cut, PME |
| `Typifier` | `typifier/` | MolGraph → typed Frame | MMFFTypifier |
| `Constraint` | `molrs-pack/constraint.rs` | Packing geometry | InsideBox, InsideSphere, OutsideSphere, Plane, Cylinder, Ellipsoid |

## Key Subsystems

### Potential System (molrs-core/src/potential/)

`KernelRegistry` maps `(category, style_name)` → `KernelConstructor`. Categories: bonds, angles, dihedrals, impropers, pairs, kspace. `ForceField::compile(frame)` resolves topology and constructs `Potentials` (aggregate sum). Coordinate format: flat `[x0,y0,z0, x1,y1,z1, ...]` (3N elements). MMFF94 parameters embedded at compile time from `data/mmff94.xml`.

### Free-Boundary Support

`SimBox::free(points, padding)` creates a non-periodic bounding box from atom positions. `NeighborQuery::free(points, cutoff)` and the WASM `LinkedCell` class auto-generate this box when no SimBox is present. RDF normalization falls back to bounding-box volume for free-boundary systems.

### Gen3D Pipeline (molrs-core/src/gen3d/)

Multi-stage 3D coordinate generation: distance geometry → fragment assembly → coarse minimization → rotor search → final minimization → stereo guards. Public API: `generate_3d(mol, opts) -> Result<(MolGraph, Gen3DReport)>`.

### Packing (molrs-pack/)

Faithful Packmol port with GENCAN optimizer. Three phases: (0) per-type sequential packing, (1) geometric constraint fitting, (2) main loop with inflated tolerance and movebad heuristic. See `.claude/skills/learn-packmol/SKILL.md` for canonical hyperparameters and mandatory Packmol-alignment workflow.

### FFI Layer (molrs-ffi/)

Handle-based design: `FrameId` from SlotMap (index + generation), `BlockHandle` with frame_id + key + version counter. Version increments on block modification for invalidation detection. No raw pointers cross FFI.

## Critical Conventions

- **Constraint gradients**: All constraints accumulate TRUE gradient (∂violation/∂x) using `+=`. Optimizer negates for descent.
- **Rotation convention**: LEFT multiplication `R_new = δR * R_old` for `apply_scaled_step`. RIGHT mult causes gradient/step mismatch.
- **Coordinate format**: Potentials use flat `[x0,y0,z0, x1,y1,z1, ...]` vectors (3N elements), not Nx3 matrices.
- **`Cell<f64>` is NOT Sync**: Use `AtomicU64` with `f64::to_bits()`/`f64::from_bits()` for interior mutability in Sync contexts.

## Development Skills & Agents

Use `/molrs-impl <feature description>` to orchestrate multi-agent feature development. Use `/molrs-spec <natural language requirement>` to generate detailed specs from requirements. See `.claude/skills/` for all available skills.
