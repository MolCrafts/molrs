---
name: molrs-arch
description: Architectural standards for the molrs Rust workspace — crate dependency rules, module ownership, trait conventions, FFI boundaries, naming. Reference document only; no procedural workflow.
---

Reference standard for molrs architecture. The `molrs-architect` agent applies these rules; this file defines them.

## Workspace Crates (8 active members)

```
molrs-core ── foundational data + algorithms
   ├── molrs-io       (file format readers/writers)
   ├── molrs-compute  (trajectory analysis: RDF, MSD, clustering, tensors)
   ├── molrs-smiles   (SMILES parser → MolGraph)
   ├── molrs-ff       (force fields, potentials, typifier)
   ├── molrs-embed    (3D coordinate generation pipeline)
   ├── molrs-pack     (Packmol-port molecular packing)
   └── molrs-cxxapi   (CXX bridge to Atomiverse C++)
```

Dirs `molrs-ffi/`, `molrs-wasm/`, `molrs-capi/`, `molrs-python/` exist on disk but are NOT workspace members — treat as inactive.

### Crate Dependency Rules (ENFORCED)

- `molrs-core` depends on no other workspace crate.
- All other crates may depend on `molrs-core` only.
- `molrs-ff` may additionally depend on `molrs-io` for parameter file parsing.
- `molrs-cxxapi` depends on `molrs-core` and `molrs-io`.
- No cyclic dependencies. A sibling crate cannot depend on another sibling unless explicitly listed above.

## Module Ownership

| Crate | Owns |
|---|---|
| `molrs-core` | `Frame`, `Block`, `Grid`, `MolGraph`, `MolRec`, `Topology`, `Element`, `SimBox`, neighbors, `math`, stereochemistry, rings, rotatable bonds, hydrogen perception, Gasteiger charges, atom-type mapping, atomistic/coarse-grain models |
| `molrs-io` | `pdb`, `xyz`, `lammps_data`, `lammps_dump`, `chgcar`, `cube`, `zarr` (Zarr V3 trajectories) |
| `molrs-compute` | `rdf`, `msd`, `cluster`, `accumulator`, `reducer`, gyration/inertia tensor, radius_of_gyration, center_of_mass |
| `molrs-smiles` | `scanner`, `parser`, `ast`, `validate`, `to_atomistic` |
| `molrs-ff` | `forcefield`, `potential` (KernelRegistry), `typifier`, `molrec_ext` |
| `molrs-embed` | `distance_geometry`, `fragment_data`, `optimizer`, `rotor_search`, `stereo_guard`, `pipeline` |
| `molrs-pack` | `constraint`, `gencan`, `objective`, `packer`, `initial`, `movebad`, `hook`, `validation` |
| `molrs-cxxapi` | CXX bridge to C++ (`bridge.rs` is build.rs-generated) |

## Trait Design Principles

1. **Object-safe**: no `Self` in return position, no generic methods on the trait.
2. **`Send + Sync` for shared trait objects**: required for rayon and cross-thread use.
3. **Owned returns at API boundary**: `eval()` returns `(F, Vec<F>)`, not borrows into self.
4. **Open-ended dispatch via registration**, not enum match: `KernelRegistry`, constraint registry, etc.
5. **Coordinate format split**:
   - Structural code: ndarray (`F3`, `FNx3`).
   - Potential/constraint kernels: flat `&[F]` (3N elements) for cache and SIMD.

## Float / Integer Precision

- `F = f64` always. The `f64` feature flag is deprecated and ignored.
- `I = i32` always. The `i64` feature flag is deprecated and ignored.
- `U = u32` always. The `u64` feature flag is deprecated and ignored.
- Neighbor list internals use `u32` directly (natural type, no alias).
- Algorithm code MUST use the `F`/`I`/`U` aliases. Raw types only for explicit bit-level work (e.g., `f64::to_bits()`).

## FFI Boundary Rules

1. **Handles only** — no `Box<T>`, `&T`, or raw pointers crossing the boundary.
2. **Version-tracked invalidation** — every block mutation increments a counter; consumers compare before use.
3. **No panics in `extern "C"` / `#[cxx::bridge]` functions.** Return error indicators (or `Result<T, JsValue>` for WASM).
4. **Naming**: `molrs_<noun>_<verb>` for C ABI (e.g., `molrs_frame_new`).
5. **Ownership stays in Rust** (SlotMap store).

## Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Crate | `molrs-<domain>` | `molrs-core`, `molrs-pack` |
| Module | snake_case domain noun | `neighbors`, `potential`, `forcefield` |
| Trait | PascalCase capability noun | `NbListAlgo`, `Potential`, `Constraint` |
| Struct (impl) | PascalCase specific noun | `LinkCell`, `MorseBondKernel` |
| Type alias | Short uppercase | `F`, `F3`, `FNx3` |
| KernelRegistry key | `("category", "style")` | `("bond", "harmonic")` |
| FFI function | `molrs_<noun>_<verb>` | `molrs_frame_new` |

## Critical Conventions

- **Constraint gradients**: accumulate TRUE gradient (∂violation/∂x) with `+=`. Optimizer negates for descent.
- **Rotation**: LEFT multiplication `R_new = δR * R_old` for `apply_scaled_step`.
- **`Cell<f64>` is NOT Sync**: use `AtomicU64` with `f64::to_bits()` / `f64::from_bits()` for interior mutability in Sync contexts.
- **Immutability**: Frame/Block operations return new instances. Mutable access only via explicit `_mut` methods.

## Compliance Checklist

- [ ] Crate dependency respects the graph above
- [ ] Module placed in the crate that owns the domain
- [ ] Trait object-safe + `Send + Sync` where shared
- [ ] Float type uses `F` alias (no raw `f32`/`f64` in algorithm code)
- [ ] Coordinate convention matches use site (ndarray vs flat `&[F]`)
- [ ] FFI handle-based, version-tracked, panic-free
- [ ] New kernels registered, not match'd
- [ ] File < 800 lines; function < 50 lines
- [ ] Public APIs return `Result`; no `unwrap()` in library code
- [ ] Naming follows the table

## Common Anti-Patterns

| Anti-Pattern | Fix |
|---|---|
| `f32` literal in algorithm | Use `F` alias |
| `Vec<Vec<F>>` for coordinates | Use `FNx3` (`Array2<F>`) |
| Raw pointer at FFI boundary | `FrameId` / `BlockHandle` |
| `match` on kernel type | Register in `KernelRegistry` |
| `Cell<f64>` in Sync struct | `AtomicU64` + `to_bits()`/`from_bits()` |
| `&mut Frame` everywhere | Return new Frame |
| `unwrap()` in library code | `?` operator, return `Result` |
| 1000+ line file | Split by responsibility |
