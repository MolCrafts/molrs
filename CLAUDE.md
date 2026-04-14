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

8 active workspace members. `molrs-core` is the foundation; everything else depends on it
(plus, in a few cases, on each other as listed):

```
molrs-core ── molrs-io ── molrs-cxxapi
            ── molrs-compute
            ── molrs-smiles
            ── molrs-ff (may also depend on molrs-io for parameter files)
            ── molrs-gen3d
            ── molrs-pack
```

| Crate | Purpose |
|---|---|
| `molrs-core` | Frame/Block/Grid/MolGraph/MolRec/Topology/Element, neighbors, math, region (SimBox), stereochemistry, rings, Gasteiger charges, hydrogen perception, atom-type mapping |
| `molrs-io` | File I/O: PDB, XYZ, LAMMPS data/dump, CHGCAR, Gaussian Cube, Zarr V3 trajectories |
| `molrs-compute` | Trajectory analysis: RDF, MSD, clustering, gyration/inertia tensors |
| `molrs-smiles` | SMILES parser → MolGraph |
| `molrs-ff` | Force fields, potentials (KernelRegistry), atom typifier |
| `molrs-gen3d` | 3D coordinate generation: distance geometry, fragment assembly, optimizer, rotor search |
| `molrs-pack` | Molecular packing: faithful Packmol port, GENCAN optimizer, geometric constraints |
| `molrs-cxxapi` | CXX bridge to Atomiverse C++ (zero-copy I/O via `FrameView`) |

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
| `Restraint` | `molrs-pack::restraint` | Packing soft-penalty (Packmol convention) | 15 concrete pub structs: `InsideBox`/`InsideCube`/`InsideSphere`/`InsideEllipsoid`/`InsideCylinder` + `Outside*` variants + `AbovePlane`/`BelowPlane` + `AboveGaussian`/`BelowGaussian` (suffix `…Restraint`); user types `impl Restraint` sit in the same type slot |
| `Region` | `molrs-pack::region` | Geometric predicate + combinators | `InsideBoxRegion`/`InsideSphereRegion`/`OutsideSphereRegion` + `And`/`Or`/`Not` combinators; `FromRegion<R>` lifts any Region to a `Restraint` via quadratic exterior penalty |
| `Relaxer` | `molrs-pack::relaxer` | Reference-geometry modification between GENCAN calls | `TorsionMcRelaxer` (`Hook` alias retained for one cycle) |
| `Handler` | `molrs-pack::handler` | Observer callbacks (on_start / on_step / on_phase_start / on_phase_end / on_inner_iter / on_finish) | `NullHandler`, `ProgressHandler`, `EarlyStopHandler`, `XYZHandler` |
| `Objective` | `molrs-pack::objective` | GENCAN's view of the packing problem (abstraction over `PackContext::evaluate`) | `PackContext` |

## Key Subsystems

### Potential System (molrs-ff/src/potential/)

`KernelRegistry` maps `(category, style_name)` → `KernelConstructor`. Categories: bonds, angles, dihedrals, impropers, pairs, kspace. `ForceField::compile(frame)` resolves topology and constructs `Potentials` (aggregate sum). Coordinate format: flat `[x0,y0,z0, x1,y1,z1, ...]` (3N elements). MMFF94 parameters embedded at compile time from `data/mmff94.xml`.

### Free-Boundary Support

`SimBox::free(points, padding)` creates a non-periodic bounding box from atom positions. `NeighborQuery::free(points, cutoff)` auto-generates this box when no SimBox is present. RDF normalization (`molrs-compute`) falls back to bounding-box volume for free-boundary systems.

### Gen3D Pipeline (molrs-gen3d/)

Multi-stage 3D coordinate generation: distance geometry → fragment assembly → coarse minimization → rotor search → final minimization → stereo guards. Public API: `generate_3d(mol, opts) -> Result<(MolGraph, Gen3DReport)>`.

### Packing (molrs-pack/)

Faithful Packmol port with GENCAN optimizer. Three phases: (0) per-type sequential packing, (1) geometric pre-fit, (2) main loop with inflated tolerance + `movebad` heuristic. See `.claude/skills/learn-packmol/SKILL.md` for canonical hyperparameters and the Packmol-alignment workflow.

**Extension points** follow direction 3 (spec `§0 bullet 9`): public trait + N concrete pub structs with own semantically-named fields; user types `impl X` identically. No `Builtin*` wrappers, no tagged-union enum in the public API, no builder pattern. Applies uniformly to `Restraint` / `Region` / `Relaxer` / `Handler`.

**Restraint vs Constraint**: Packmol's "constraints" (InsideBox, Sphere, Plane, Ellipsoid, Cylinder, Gaussian) are implemented as `scale * max(0, d)` / `scale2 * max(0, d)²` soft penalties — honest name is `Restraint`. No hard-constraint (Lagrange / SHAKE / RATTLE) machinery exists in molrs-pack; the `Constraint` trait is reserved for future work.

**Scope equivalence law** (spec `§4`): `Molpack::add_restraint(r)` broadcasts to every target, semantically equivalent to `for t in targets { t.with_restraint(r.clone()) }` — no separate global-storage path in `PackContext`.

### FFI Layer (molrs-cxxapi/)

CXX bridge to Atomiverse C++. Zero-copy I/O via `FrameView` (borrowed) into existing `write_xyz_frame`; owned `Frame` only built when persisting to MolRec (Zarr). Bridge generated from `#[cxx::bridge]` in `bridge.rs`. No raw pointers cross the boundary.

## Critical Conventions

- **Restraint gradients**: `Restraint::fg` accumulates the TRUE gradient (∂penalty/∂x) INTO `g` with `+=` (never overwrite). Optimizer negates for descent.
- **Two-scale contract**: linear penalties (box / cube / plane — Packmol kinds 2/3/6/7/10/11) consume `scale`; quadratic penalties (sphere / ellipsoid / cylinder / gaussian — kinds 4/5/8/9/12/13/14/15) consume `scale2`. Each `impl Restraint` picks one internally.
- **Rotation convention**: LEFT multiplication `R_new = δR * R_old` for `apply_scaled_step`. RIGHT mult causes gradient/step mismatch.
- **Coordinate format**: Potentials use flat `[x0,y0,z0, x1,y1,z1, ...]` vectors (3N elements), not Nx3 matrices.
- **`Cell<f64>` is NOT Sync**: Use `AtomicU64` with `f64::to_bits()`/`f64::from_bits()` for interior mutability in Sync contexts.

## Development Skills & Agents

Skills (the WHAT — reference standards) and agents (the HOW — active executors) are paired
by domain. Skills define the rules; agents apply them.

| Domain | Skill | Agent |
|---|---|---|
| Architecture | `molrs-arch` | `molrs-architect` |
| Performance | `molrs-perf` | `molrs-optimizer` |
| Documentation | `molrs-doc` | `molrs-documenter` |
| Testing | `molrs-test` | `molrs-tester` |
| Scientific correctness | `molrs-science` | `molrs-scientist` |
| FFI safety | `molrs-ffi` | (inline review against the skill) |

Standalone skills (no paired agent):

- `/molrs-impl <feature>` — single entry point for feature work; orchestrates the agents above
- `/molrs-spec <requirement>` — convert NL requirement to a spec doc (also invoked internally by `/molrs-impl` Phase 0)
- `/molrs-review [path]` — aggregate review across all dimensions
- `/molrec-compat <format>` — evaluate molrec compatibility for a format
- `/learn-packmol` — Packmol Fortran reference discipline

See `.claude/skills/` and `.claude/agents/` for the full text.
