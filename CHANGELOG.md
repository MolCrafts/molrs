# Changelog

All notable changes to molrs are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

### Added

- **Whole-chain acceptance for `chem-perceive` (16 specs).** Two new test targets
  verify the *chain*, which no spec on it ever had — each verified its own slice.

  - `molrs/tests/architecture_gate.rs` — the five "only one of these exists"
    promises, as machine-checked gates, none of which can exempt itself (every
    needle is assembled with `concat!`, so the string a gate searches for does not
    occur in the gate): one home and one form for parameters (flat `ff/params/`, no
    `include_str!`, no runtime parse of a **built-in** table — told from a *user's*
    XML by the signature, since a function that parses text it was never given is
    parsing a table molrs shipped as a string); one perception layer; one
    interpolation seam (`ParameterInterpolator` has exactly one implementor in
    `src/`); one MMFF path (no bespoke energy layer, no second classifier, no
    MMFF-owned kernel); and **a registered kernel constructor that ignores `tp` is
    not a Style** — now decided on *semantics* (does the body read the binding?)
    rather than on the *spelling* of the binding, which is what let `pme_ctor` and
    `pair_coul_cut_ctor` through before.
  - `molrs/tests/end_to_end.rs` — SMILES/SDF → Perceive → AtdTypifier → ChargeModel
    → ForceField → Potentials → E + F, on **all 37** antechamber molecules and
    **all 11** RDKit MMFF fixtures, against oracles molrs did not produce. Every
    fixture list is directory-scanned and every partition (zero-charge,
    delocalized-N) is a **predicate evaluated on the molecule**, never a list of
    names.
  - `molrs-python/tests/test_parity.py` — the bindings lose no precision: `float64`
    end to end, charge conservation to **1e-12**, no renormalization, and every
    bit-level invariant the Rust suite asserts.

### Fixed

- Nothing. This acceptance repaired nothing it found, by design: *an acceptance that
  quietly repairs what it finds is the last place a defect can hide, because the thing
  that would have reported it is the thing that swallowed it.* Three gates are RED on
  landing, each naming a real defect, and each gets its own spec.

### Known defects (found by this acceptance, deliberately NOT fixed here)

- **GAFF/GAFF2 force fields declare no electrostatic style.**
  `gaff_forcefield` builds `atom/full`, `pair/lj/cut` and the bonded styles — and no
  Coulomb style at all. `to_potentials(...).calc_energy_forces(...)` therefore returns
  an energy with **zero electrostatic contribution, silently**, for every molecule,
  including the ionic ones (acetate, methylammonium, imidazolium). This is the caffeine
  hole (150 kcal/mol on the MMFF path) reproduced in the other force field, and no
  stage test could see it because no test ran the GAFF chain to an *energy*. The tell
  was already in the tree: `SpecialBonds.coul = [0, 0, 1/1.2]` — AMBER's SCEE, a 1-4
  Coulomb scale factor declared for a term that does not exist.
  Gate: `end_to_end::the_force_field_the_chain_builds_declares_its_electrostatics`.
- **Four test-tree subset assertions** over the 11 MMFF energy fixtures
  (`mmff/energy.rs::S_NAMES`, the zero-charge pair, `typifier/mmff_variant.rs::
  N_FIXTURES` / `IDENTICAL_FIXTURES`). Each is a hand-written list where the predicate
  is computable, and the gap is not hypothetical: `e_caffeine` and `e_big` **do** carry
  a delocalized nitrogen and appear in neither list, so nothing asserted that MMFF94s
  changes their energy at all.
  Gate: `architecture_gate::no_test_asserts_on_a_subset_of_its_fixtures`.
- **One test goes green by skipping itself** when its input is absent
  (`ff/readers/opls.rs::reads_real_molpy_oplsaa`, which needs a molpy checkout). In CI
  it asserts nothing and counts as coverage.
  Gate: `architecture_gate::no_test_returns_green_when_its_input_is_absent`.

- **A single live graph-reference layer for Python.** `molrs.views` now owns
  `NodeRef`, `RelationRef`, weak interning, live property mappings, ref
  collections, and the concrete `Atom` / bonded-term / `Bead` types. Refs are
  always bound to a graph handle; graph factories create the handle before the
  Python object.

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

  Two reasons, and the second is why this is a removal rather than a deprecation.

  *It was a duplicate.* The generic path reproduces RDKit on all 11 MMFF fixtures
  (worst deviation 3.1e-9 kcal/mol) and matches a frozen per-style breakdown on
  every one of its 7 energy terms to 1e-6 — so the bespoke layer was a second
  implementation of the same force field, i.e. a second set of numbers to be wrong.

  *It had already drifted.* Until the preceding fix, `build()` compiled potentials
  with **no electrostatic style at all** (no `ForceField` defined `pair/mmff_ele`):
  caffeine was ~150 kcal/mol low. The shortcut is what hid it — by folding
  typify → compile into one call it removed the `Frame` where a missing term is
  visible. The three `typify_*` classifiers were wrong on their own terms too: an
  aromatic bond came back as bond type 1 (RDKit says 0), and `typify_angle(bt_ij,
  bt_jk)` could not return 3 for a cyclopropane C-C-C angle *at any input*, because
  ring membership was not among its arguments.

  **Downstream: `molpack`** consumes `molcrafts-molrs` natively and its relaxer
  follows the `build`-style pattern documented in `docs/interop.md`; it must move to
  the `typify` → `to_potentials` route. This is the only known external consumer.

- **The four MMFF parameter files lose 4,065 type-def rows.** `<Bond>` (493),
  `<Angle>` (2342), `<StretchBend>` (282), `<Torsion>` (926) and `<Oop>` (117) are
  gone from `data/mmff94{,s}.xml` and `molrs/data/mmff94{,s}.xml`, along with the
  five readers that parsed them. **No code ever read one of them.** MMFF's bonded
  parameters depend on aromaticity, ring size, four-level equivalence degradation
  and empirical fallbacks — they are not a `(type_i, type_j, …) → params` table —
  so the typifier resolves them per interaction and bakes them into Frame columns,
  which is what the kernels have always read. The rows existed solely to satisfy a
  guard requiring every style to carry type definitions. `<VdW>` keeps all 95 rows
  (van der Waals genuinely *is* a per-atom-type table) and `<ElectrostaticParams>`
  stays. A parameter file is unaffected unless it defined those sections.

### Added

- **`ParamSource`** (`molrs::ff::potential::ParamSource`) names where a kernel gets
  its parameters: `TypeRows` (from the style's type-def rows) or `PerInstance` (from
  Frame columns the typifier baked). `KernelRegistry::register_with` /
  `register_kernel_with` carry it; the plain `register` / `register_kernel` remain as
  the `TypeRows` short form, so no existing kernel registration changes.
  `Style::to_potential`'s empty-type-params guard now consults it — a `TypeRows`
  style with no rows still errors, a `PerInstance` style is allowed zero rows — which
  replaces the old blanket `category != "pair"` escape hatch. Eight styles are
  registered `PerInstance`: MMFF's `bond/mmff_bond`, `angle/mmff_angle`,
  `angle/mmff_stbn`, `dihedral/mmff_torsion`, `improper/mmff_oop`, `pair/mmff_ele`,
  plus `pair/coul/cut` and `kspace/pme` (both read per-atom charge off the Frame).
  `pair/mmff_vdw` stays `TypeRows`. The invariant — *a registered kernel constructor
  that ignores its type-params is not a Style* — is enforced in both directions by
  `tests/ff/potential/param_source_gate.rs`.

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
