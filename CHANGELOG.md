# Changelog

All notable changes to molrs are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

### Added

- **MMFF94s is reachable from the typifier / force-field path.** New
  `MMFF94STypifier` (Rust: `molrs::ff::typifier::mmff::MMFF94STypifier`; Python:
  `molrs.MMFF94STypifier`) types and parameterises a molecule with the MMFF94s
  ("static", Halgren 1999) parameter set, giving `data/mmff94s.xml` its first
  consumer. MMFF94 and MMFF94s share all 95 atom types and every bond / angle /
  stretch-bend / vdW / charge parameter; they differ in 11 out-of-plane rows and 42
  torsion rows, all centred on delocalised trivalent nitrogen (MMFF types 10 `NC=O`
  / 40 `NC=C`), which MMFF94s flattens by raising `koop` to `+0.015` / `+0.030`
  md·Å·rad⁻².

### Fixed

- **The MMFF typifier no longer hardcodes the MMFF94 variant.**
  `frame_builder::annotate_mmff` now threads the variant into
  `MmffMolProperties::compute`, `torsion_params` and `oop_koop`, so the `koop` and
  `(v1, v2, v3)` columns baked onto the typed Frame — the ones the `mmff_oop` /
  `mmff_torsion` kernels actually read — follow the typifier the caller chose. The
  MMFF94s tables (`MMFF_OOP_S`, `MMFF_TOR_S`) had been shipped and used only by the
  bespoke energy path.

### Changed — BREAKING

- **`MMFFTypifier` is renamed to `MMFF94Typifier`** (Rust and Python), and its
  fallible `MMFFTypifier::mmff94() -> Result<Self, String>` constructor is replaced
  by the infallible `MMFF94Typifier::new() -> Self` (the parameter set is embedded
  at compile time; `from_xml_str` still returns `Result`). There is **no compat
  alias and no deprecated shim**: MMFF is now two named front doors —
  `MMFF94Typifier` and `MMFF94STypifier` — over one engine, and the variant is a
  private field, never a constructor flag. Migration:
  `MMFFTypifier::mmff94()?` → `MMFF94Typifier::new()`; `molrs.MMFFTypifier()` →
  `molrs.MMFF94Typifier()`.

## [0.7.0] - 2026-07-08

molrs and `molcrafts-molpy` continue to share one version line and release as a
pair; downstream exact-pins move to `0.7.0`.

### Added

- **`molrs.typifier` Python submodule.** `MMFFTypifier` and `OPLSAATypifier` are
  exposed as a first-class `molrs.typifier` module (also re-exported at the
  `molrs` top level), giving the Python bindings a native
  typify → typed `Atomistic` → `Frame` path. molpy's OPLS-AA / MMFF typifiers are
  now thin re-exports of these.
- **Daylight reaction-SMARTS (SMIRKS) transform engine** (`chem`) — apply
  reaction templates as graph edits; `Reaction.apply` returns handles to the
  atoms it touched.
- **Atom-map SMARTS matcher and graph-edit conveniences** exposed to Python.
- **Isomorphism-invariant structural graph hash on `MolGraph`** — a canonical
  hash stable under atom re-indexing, backing region-scoped retype caching in
  downstream consumers.
- **Frame serialization foundation** — the `serde` feature adds
  `Serialize`/`Deserialize` for `Frame`/`Block`/`Column`/`SimBox`, and the
  `stream` feature adds MessagePack/JSON wire encoding
  (`frame_to_bytes`/`bytes_to_frame`). WASM-clean; intentionally not in `full`.
  (The WebSocket streaming/control layer on top of this is deferred; see the
  `net-streaming` spec.)

## [0.6.0] - 2026-07-03

molrs and `molcrafts-molpy` continue to share one version line and release as a
pair; downstream exact-pins move to `0.6.0`.

### Changed (breaking)

- **Freud-style `compute` reorganization.** Trajectory analyses are grouped into
  per-category folders — `transport`, `spectroscopy`, `shape`, `ml`, `dynamics`,
  `dielectric` — replacing the flat per-analysis modules (`van_hove`, `pca`,
  `kmeans`, `gyration_tensor`, `inertia_tensor`, `onsager`, `spectra`, …).
  **Crate-root re-exports of every relocated type are preserved**
  (`molrs::compute::VanHove`, `::KMeans`, `::OnsagerCorrelation`,
  `::CenterOfMass`, … still resolve); only fully-qualified
  `compute::<old_submodule>::` paths change.

### Removed (breaking)

- **Native GAFF/AMBER parameter estimator.** `ParameterEstimator`, the
  `ff::typifier::estimate` module, and its embedded `gaff_empirical.json` /
  `gaff_equiv.json` tables are removed; estimating parameters for uncovered
  bonded terms is delegated to embedded reference. `OplsTypifier::with_estimator` /
  `with_default_estimator` and the Python `OplsTypifier(estimator=…)` argument
  are gone. The force-field-agnostic `Estimator` trait + `assign_bonded_with`
  seam is **kept** as an extension point (molrs ships no in-tree estimator).
- `compute::jacf::{JacfResult, green_kubo_conductivity}` — the Green–Kubo
  conductivity is now `GreenKuboConductivity` (raw ACF) composed with
  `fit::RunningIntegral`; `jacf` is a documentation-only module.
- Python top level: `molrs.MolRec` and `molrs.Observables` (the
  `ScalarObservable` / `VectorObservable` types are kept).

### Added

- **Streaming accumulators** `RDFAccumulator`, `MSDAccumulator`,
  `VACFAccumulator` — bounded-memory online accumulation for long trajectories
  (Rust API).

## [0.5.1] - 2026-07-01

Version realigned with `molcrafts-molpy`: molrs and molpy now share one version
line and are released as a pair (jumped 0.1.6 → 0.5.1 to converge).

### Added

- **analysis-parity compute suite (Python-exposed).** Geometric distribution
  functions (ADF / DDF / distance), combined distribution functions (CDF),
  spatial distribution function (SDF), Van Hove `G(r, t)` (self + distinct) and
  Legendre reorientational TCFs, geometric hydrogen-bond detection, native
  periodic radical (Laguerre) Voronoi tessellation with domain/void analysis and
  electron-density integration, and vibrational spectra (VDOS / IR / Raman /
  VCD / ROA / resonance Raman) via the time-correlation route.
- **Cube trajectory reader** for *ab initio* electron-density trajectories.

## [0.1.6] - 2026-06-21

No library changes — tooling/CI only (no Rust, Python, or WASM API surface change).

### Removed

- Stop version-controlling lockfiles (`Cargo.lock`, `uv.lock`); CI and the git
  hooks resolve dependencies fresh (dropped `cargo --locked`).

### Changed

- CI: the benchmark workflow publishes perf history only on the canonical
  `MolCrafts/molrs` repo (forks lack the gh-pages branch).

## [0.1.5] - 2026-06-20

### Changed (breaking)

- **Force-field angles are now radians internally.** Angle `theta0`, dihedral/
  improper phase `d`/`phi`/`chi0` are stored and consumed in radians; kernel
  constructors no longer call `.to_radians()`, and each reader normalizes
  user-facing degree input at its boundary (LAMMPS `*.ff` deg→rad, MMFF94 XML
  deg→rad; the OPLS/GROMACS reader was already radians). Fixes a double-conversion
  that produced ~100+ kcal/mol of spurious angle energy on OPLS-typified
  structures at their reference angle.

## [0.1.4] - 2026-06-18

### Added

- **GROMACS TRR and XTC trajectory I/O.** Native readers and writers for the
  `.trr` (full-precision XDR; single/double; coordinates, velocities, forces)
  and `.xtc` (XDR + lossy `xdr3dfcoord` compression; classic 1995 and 2023
  magic) formats, alongside the existing DCD/GRO support. Each exposes
  sequential reads, single-frame access, and O(1) random access via
  `TrajReader::read_step` (lazy per-frame offset index), plus writers. The XTC
  compression codec is a clean-room implementation. Surfaced in Python as
  `molrs.read_trr`/`read_xtc`, the lazy `molrs.io.read_trr_trajectory`/
  `read_xtc_trajectory`, and `write_trr`/`write_xtc`.
- **compute ↔ fit separation.** Trajectory `Compute`s return raw curves/ACFs; a
  separate `Fit` family (`compute::fit`) performs the numerical fitting and
  spectral transforms as an explicit downstream step.

### Changed

- **Packaging: single published crate.** The former seven workspace crates
  (`core`/`io`/`signal`/`compute`/`ff`/`conformer` + façade) are merged into one
  `molcrafts-molrs` with feature-gated modules. The public Rust, Python, and
  WASM API surfaces are unchanged.

## [0.1.3] - 2026-06-14

### Added

- **Force-field FFI handle.** `molrs_ffi::ForceFieldRef` — a stable, zero-copy
  handle for `molrs::ff::ForceField` (the force-field analogue of `FrameRef`),
  gated behind a new `ff` feature on `molcrafts-molrs-ffi`. Lets force-field
  consumers (e.g. molpack) borrow a `ForceField` across the FFI boundary.
- **`molrs::ff::potential::intramolecular_pairs`** — builds the intramolecular
  neighbour-pair `Block` for a frame, consumed by relaxation / energy callers.
- LAMMPS force-field reader and per-instance force-field parameter support.

### Changed

- `molrs::ff::potential` is now a directory module (`potential/`) rather than a
  single file; MMFF typifier internals reorganized. The WASM / TypeScript
  public surface (`@molcrafts/molrs` npm) is unchanged.

## [0.1.0] - 2026-06-10

### Added

- `Frame.from_dict` on the native PyO3 core — accepts either the
  `{"blocks": {...}, "metadata": {...}}` envelope or a direct
  `{name: {column: array}}` mapping, completing the `to_dict` / `from_dict`
  round-trip. Column values use the same accepted types as `Block.insert`.
- `molrs.BlockDtypeError` — public exception (subclasses `TypeError`) raised by
  `Block.insert` on a non-numpy-representable column. Importable and stable so
  downstream code can `except molrs.BlockDtypeError` precisely.

### Changed

- **Block column dtype contract is now numpy-only, fail-fast (behavior change).**
  `Block.insert` (and therefore every `Frame` column write) now accepts only
  numpy-representable dtypes — float, int, bool, and str. Object-dtype,
  None-bearing, and ragged/mixed arrays were previously rejected with a generic
  `TypeError`; they now raise the new public `molrs.BlockDtypeError` with a
  message naming the offending column and the detected dtype. There is no
  Python-side object-column overflow — columns the Rust Store cannot represent
  must be coerced to a supported dtype or dropped by the caller.
