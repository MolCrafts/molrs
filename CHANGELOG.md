# Changelog

All notable changes to molrs are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

## [0.8.0] - 2026-07-17

molrs and `molcrafts-molpy` continue to share one version line and release as a
pair; downstream exact-pins move to `0.8.0`. This minor is the **chem-perceive**
line: chemical perception, force-field ownership, and whole-chain acceptance.

### Added

- **Chemical perception + GAFF/ATD/AM1-BCC pipeline.** Rings, aromaticity,
  hydrogens, stereo, BCC bond types, charge equivalencing, Gasteiger, and the
  full ATD typifier (seven parameter tables) land as always-on `perceive` plus
  `ff` consumers. Antechamber / GAFF / GAFF2 parameter tables are committed
  Rust (`ff/params/`), not runtime-parsed text. `Parmchk2Estimator` is the
  single interpolation seam for missing bonded terms.
- **A native nominal `Typifier` base and batch reaction execution.** Native
  OPLS-AA, MMFF94/MMFF94s, and ATD typifiers inherit the same subclassable
  Python base used by downstream typifiers. `Reaction.apply_many` resolves all
  leaving groups against the intact graph, removes their union in one relation
  scan, and applies every disjoint transform with one touched set per binding;
  rooted SMARTS matching reuses one molecular context across the batch.
- **Native ownership for the remaining `molpy.core` kernels.** The Python
  bindings now expose the Rust-owned `Element`, units and LJ reduced
  units, free-boundary neighbor queries, complete Box batch geometry and
  conversions, subclassable/composable regions, graph direction alignment and
  replication, plus CL&Pol fragment scaling/scaleLJ with a compiled parameter
  table. These APIs let molpy consume the owner types directly and retain only
  Python call-shape sugar around higher-level workflows.
- **A single live graph-reference layer for Python.** `molrs.views` owns
  `NodeRef`, `RelationRef`, weak interning, live property mappings, ref
  collections, and the concrete `Atom` / bonded-term / `Bead` types. Refs are
  always bound to a graph handle; graph factories create the handle before the
  Python object.
- **MMFF94s is reachable from the typifier / force-field path.** New
  `MMFF94STypifier` (Rust + Python) types with the MMFF94s ("static", Halgren
  1999) set. MMFF94 and MMFF94s share atom types and most parameters; they
  differ on out-of-plane / torsion rows centred on delocalised trivalent
  nitrogen (types 10 / 40).
- **`ParamSource`** (`molrs::ff::potential::ParamSource`) names where a kernel
  gets parameters: `TypeRows` or `PerInstance`. Eight styles register
  `PerInstance` (MMFF bonded + `pair/mmff_ele`, plus `pair/coul/cut` and
  `kspace/pme`). The invariant *a registered kernel constructor that ignores
  `tp` is not a Style* is enforced in both directions.
- **Whole-chain acceptance for `chem-perceive`.** Architecture gates
  (`architecture_gate.rs`) and end-to-end SMILES/SDF → Perceive → Typifier →
  ChargeModel → ForceField → Potentials on all antechamber + RDKit MMFF
  fixtures (`end_to_end.rs`), plus Python bit-level parity
  (`molrs-python/tests/test_parity.py`). Acceptance does not silently repair
  defects it finds — each defect became its own fix (below).

### Changed — BREAKING

- **`Element` has one owner and one public Rust path: `molrs::Element`.** The
  implementation module is private; CXX generates `molrs::Element` from that
  canonical enum, and Python exposes the same 1–118 domain. `ElementData`,
  `Element.initialize()`, atomic number 0, `X`, unknown-element property
  defaults, and downstream aliases are deleted. Invalid values fail instead
  of constructing a sentinel element.
- **Frame exchange is schema version 2 only.** A Frame is blocks + exact typed
  `meta` + `simbox`; the 19 `MetaValue` dtypes keep their scalar/vector type
  across Rust, C, CXX, Python, Atomiverse, and molpy. String metadata,
  `frame.box`, alternate metadata payloads, and every pre-v2 decoder are
  deleted. Non-v2 data is rejected and must be regenerated.
- **`MMFFTypifier` is renamed to `MMFF94Typifier`** (Rust and Python), and its
  fallible `MMFFTypifier::mmff94() -> Result<Self, String>` constructor is
  replaced by the infallible `MMFF94Typifier::new() -> Self`. No compat alias.
  Migration: `MMFFTypifier::mmff94()?` → `MMFF94Typifier::new()`;
  `molrs.MMFFTypifier()` → `molrs.MMFF94Typifier()`.

### Fixed

- **GAFF/GAFF2 now declare unbuffered Coulomb electrostatics.**
  `gaff_forcefield` registers `pair/coul/cut` with AMBER's measured conversion
  factor (`332.05221729` kcal·Å·mol⁻¹·e⁻²), vacuum dielectric, and `delta = 0`,
  so ionic molecules no longer evaluate with silent zero electrostatic energy.
  Gate: `end_to_end::the_force_field_the_chain_builds_declares_its_electrostatics`.
- **MMFF energy / variant tests no longer use hand-written fixture subsets.**
  Zero-charge and type-10/40 partitions are computed from every fixture on disk;
  caffeine and `e_big` can no longer sit outside the assertion that would catch
  a regression.
- **External OPLS reader test is `#[ignore]` instead of skip-and-pass.** Missing
  molpy `oplsaa.xml` no longer counts as a green assertion in CI.
- **The MMFF typifier no longer hardcodes the MMFF94 variant.**
  `frame_builder::annotate_mmff` threads the variant into
  `MmffMolProperties::compute`, `torsion_params` and `oop_koop`, so baked
  `koop` / torsion columns follow the typifier the caller chose.

### Removed — BREAKING

- **Detached node/relation state is not part of the view model.** Callers create
  nodes and relations through an owning graph factory; refs never carry a
  pending Python dictionary that later attaches to a graph.
- **The bespoke MMFF energy path is deleted. MMFF is no longer a special case.**
  `MmffForceField`, `MmffEnergyBreakdown` and the whole `ff/mmff/energy/` assembly
  layer are gone, together with every shortcut that reached them:

  | Removed | Replacement |
  |---|---|
  | `molrs::ff::mmff::MmffForceField` / `MmffEnergyBreakdown` | the generic kernels (`ff::potential::{bond,angle,dihedral,improper,pair}::mmff`) |
  | `MMFF94Typifier::build(&mol)` / `MMFF94STypifier::build(&mol)` | `typify` → `to_frame` → `intramolecular_pairs` → `ForceField::to_potentials` |
  | `molrs.build_mmff_potentials(mol)` (Python) | the same route (see below) |
  | `MMFF94Typifier::typify_bond` / `typify_angle` / `typify_dihedral` | nothing — they were **wrong**; the type codes are on the typed Frame |

  ```python
  # before
  pots = molrs.build_mmff_potentials(mol)          # or MMFF94Typifier().build(mol)

  # after
  typifier = molrs.MMFF94Typifier()
  frame = typifier.typify(mol).to_frame()          # labels + charges
  frame["pairs"] = molrs.intramolecular_pairs(frame)
  pots = typifier.forcefield().to_potentials(frame)
  ```

  **Downstream: `molpack`** must move to the `typify` → `to_potentials` route.
- **The four MMFF parameter files lose 4,065 type-def rows.** `<Bond>`,
  `<Angle>`, `<StretchBend>`, `<Torsion>`, and `<Oop>` type-def sections are
  gone from the MMFF XML tables and their readers. Bonded MMFF parameters are
  per-instance on the Frame; only `<VdW>` and `<ElectrostaticParams>` remain as
  type-row tables.

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
