# Changelog

This page summarizes recent repository history for documentation readers. The
authoritative release history remains the Git tags and GitHub releases.

## 0.1.2

Maintenance release. No public API or behavior changes — `0.1.2` is a
drop-in replacement for `0.1.1`.

- **Documentation accuracy.** Corrected developer-facing docs to match the
  shipped code: `Topology` is a native adjacency structure (no petgraph; petgraph
  is pulled in only by the SMARTS VF2 matcher under the `smiles` feature), the
  force-field entry point is `ForceField::to_potentials(frame)` (the old
  `compile` name is gone), and the CXX bridge to Atomiverse crosses coordinates
  as `f64` unconditionally (there is no `cfg!(feature = "f64")` precision switch).
- **Workspace version bump.** The merged crate and all binders
  (`molrs-ffi`, `molrs-python`, `molrs-wasm`, `molrs-capi`, `molrs-cxxapi`)
  move to `0.1.2` in lockstep.

## 0.1.0

This release makes molrs the single source of truth for the molecular force-field,
topology, and field-name model that molpy previously carried, alongside a workspace-wide
dependency refresh and module reorganization. It is the largest release since the
unified-packaging series and contains breaking changes for crate users (see below).

- **Native force-field model.** The force-field is now owned in `molrs-ff`: native
  `Style`/`Type` state with `to_potentials` returning frame-free, deferred potentials
  that bind topology and coordinates at evaluation time. A real `KernelRegistry` replaces
  the old closed match, with new bond/angle/dihedral/improper/pair kernels ported and
  finite-difference-validated (class2, morse, charmm, periodic/fourier, multi/harmonic,
  cvff, Buckingham, lj/class2, Thole, OPLS 4-cosine, coul/cut, coul/tt). An OPLS-AA /
  GROMACS force-field XML reader and an LBFGS-backed geometry optimizer round out the
  surface, all exposed through a subclassable, chainable `PyForceField` builder.
- **Canonical Frame/Block.** The rich `Frame`/`Block` layer became canonical
  `molrs.Frame`/`molrs.Block` with a numpy-only column contract that fails fast via
  `BlockDtypeError` (scalar and non-numpy columns are rejected rather than silently
  dropped). `molrs.io` readers now return these rich frames, `Frame.from_dict`/`to_dict`
  live on the native core, and `Box.is_free` / a `cell_defined` no-cell box state were added.
- **Single field registry.** `molrs-core/src/keys.rs` is now the one source of canonical
  field names (positions, velocities, residue and relation-endpoint keys), exposed as
  `molrs.keys` so the Rust and Python layers cannot drift.
- **Topology kernels for molpy sinks.** Bond-graph BFS distances, relation-id enumeration
  with scale-geometry, `ForceField.subset` projection, angle/dihedral perception, and a
  petgraph-backed `Topology` now serve molpy's graph queries directly.
- **Unit system & constants.** A new pint-style unit-conversion module (CODATA-2018/SI-2019
  registry, dimension algebra, parser, quantities) lands in `molrs-core`. Physical
  constants were hoisted to single homes — `COULOMB_REAL`/`BOLTZMANN_REAL` in core units and
  an RDKit-matched MMFF family in `molrs-ff/constants.rs`, keeping `COULOMB_REAL` (CODATA)
  and `COULOMB_MMFF` (RDKit) deliberately distinct.
- **Trajectory & misc.** The `Trajectory` pyclass is now subclassable; an axis-aligned
  `Cuboid` region primitive was added; dielectric-spectrum validation physics sank into
  `molrs-compute`; and duplicated BFS / `wrap_index` helpers were deduplicated.
- **Dependencies & build.** Upgraded across the workspace (petgraph 0.8, rand 0.10,
  criterion 0.8, rayon 1.12, wasm-bindgen 0.2.123) with MSRV raised to 1.91, the rand 0.10
  `RngExt` API migration, `--locked` lockfile enforcement on cargo build/test/clippy in CI,
  and a domain-grouped reorganization of the flat `molrs-core` and `molrs-io` modules.

Crate users: several `molrs-ff` APIs were renamed or removed — `Potential::eval` became
`calc_energy_forces` / `calc_energy` / `calc_forces`, `ForceField::compile` became
`to_potentials` (with `Style::to_potential`), the free `minimize` / `minimize_batch`
functions gave way to an LBFGS optimizer class, and parameter names were unified
(`k0` → `k` / `theta0`). Combined with the domain-grouped module reshuffle in
`molrs-core` / `molrs-io` (new `store/`, `system/`, `chem/`, `spatial/`, `data/`,
`trajectory/` paths) and the MSRV bump to 1.91, downstream code will need import-path and
call-site updates.

## 0.0.15

Version 0.0.15 continued the unified packaging work, including publish workflow
cleanup and formatting across the I/O crate.

## 0.0.12

The release workflow was hardened for WebAssembly and facade publishing. Python
publishing now passes interpreter discovery through the maturin container path.

## 0.0.11

Compute internals moved behind a unified compute DAG, and Python and WASM
bindings were adapted to the newer analysis shape.

## 0.0.10

RDF behavior was aligned more closely with freud-style normalization, and PCA
plus k-means analysis reached the public surface.

## 0.0.8

The facade crate was introduced so Rust users can depend on one public package
and opt into subsystems through Cargo features.
