# Release — molrs before molpy

## Rule

1. Land on **master**, tag **`vX.Y.Z`**, wait for **Publish** (crates.io + npm + PyPI including Pyodide wheel).
2. Only then bump **molpy** to the same **major.minor** and tag.
3. Shared pin: consumers use `molcrafts-molrs>=X.Y.0,<X.(Y+1)`.

## Publish (tag push)

Workflow `.github/workflows/publish.yml`:

| Job | Registry |
|-----|----------|
| `publish-molrs` | crates.io (`molcrafts-molrs`) |
| `publish-wasm` | npm (`@molcrafts/molrs`) |
| `build-python` | desktop wheels → artifact |
| `build-python-pyodide` | Emscripten/Pyodide wheel → artifact |
| `publish-python` | PyPI (all wheels, trusted publishing) |

Re-run the failed tag workflow, or dispatch Publish against the same tag
(idempotent skips). Branch dispatches run CI and build artifacts without
publishing. Tags must match the root package version and be on master.
Registry publications wait for CI. The public checklist is [docs/releasing.md](../../docs/releasing.md).

**Expected `cargo package` warning.** `molrs/Cargo.toml` excludes `examples/`
(the OPLS generator is a repository tool, not published), so `cargo package`
and `cargo publish` print

```text
warning: ignoring example `gen_opls_params` as `examples/gen_opls_params.rs` is not included in the published package
```

That is the exclusion working, not a packaging failure. Any *other* warning is
still a finding; `cargo package --list -p molcrafts-molrs` must list no
`examples/` path.

## scripts/

Fixture fetching and the optional shared-library verification scripts live
here. No publish helper scripts; publishing stays in the workflow.

## v0.12.1 (2026-08-05)

- SMILES/SMARTS emit: `write_smiles` / `from_atomistic` / `write_smarts` (io surface only)
- smiles-emit-01..04 closed

## v0.13.1 (2026-08-13)

Patch on the 0.13 line. Land on master, tag `v0.13.1`, wait for Publish, then molpy 0.13.1.

- `Frame.meta` is a write-through `FrameMeta` mapping: assign `frame.meta["timestep"] = 0` (plain Python scalars). `MetaValue` remains for explicit typed writes.
- Python `Block` / `Frame` columns accept `molrs.keys.Key` as well as `str`.
- NeighborList / Neighbors binders (Python + WASM) and DRS correlators already on this line since 0.13.0.

## v0.12.2 (2026-08-05)

- Public Python names only: `write_smiles` / `write_smarts` (removed `write_local_smarts` export)

## v0.13.0 (2026-08-09)

- `stream::Publisher` (was `FrameServer` / `FramePublisher`); serialization routed through `io`
- Structure and force-field readers/writers for the molpy sink
- Publish matrix: 3 OS × Python 3.12/3.13/3.14; split macOS arm64/x64 wheels

## v0.13.1 (2026-08-13)

- NeighborList engine + Neighbors table (Python / WASM)
- `molrs.keys` as Key-typed constants
- DRS correlators; shared FFT/flux primitives
- `Frame.meta` is dict-like (`FrameMeta`)

## v0.13.2 (2026-08-19)

- DCD / XTC / TRR on the `FrameIndexBuilder` streaming surface

## v0.14.0 (untagged — stay on `dev`; tag is 08)

- `UnitPreset` / `UnitPresetRegistry` in `core::units`; zero unit conversion inside MD
- `Potential` and `Compute` as `runtime_checkable` Protocols (Python)
- `MD(dtype=)` experimental (`import molrs.md` emits `FutureWarning`)
- Public record API is `Record` / `Trajectory.read` / `Trajectory.write` (not `MolRec` / `read_zarr`)
- `frame.meta` is a live write-through mapping of **plain** Python values (`FrameMeta`), not a `dict` snapshot of `MetaValue` boxes. The dtype belongs to the key: writing a plain value to an existing key keeps that key's dtype and refuses one it cannot hold, so `m[k] = m[k]` is an identity; assign a `MetaValue` to give a key a different dtype and read the tag back with `meta.dtype(k)`. A JSON document is returned decoded, so nested edits are read-modify-write. `None` stores as JSON null (it used to raise). `Frame.from_dict` is removed — `Frame(blocks=..., meta=...)` covers it, and `dict(frame.meta)` round-trips.
- One LJ pair kernel; `VerletSkin::pairs_at` is the only MIC site; PME as pair style `coul/long/pme`
- Identity scalar `Idx = u64` (retired `U = u32`); column storage widths preserved (no f32→f64 / i64→i32 / u64→u32 narrowing)
- WASM domain-uint columns are `BigUint64Array`; JS names stay `setColU32` / `copyColU32` / `viewColU32` / `hasU32`
- wasm `NeighborQuery` symmetry deferred to 0.15 (binder-surface-symmetry note)
- `Region` trait gains `distance` / `distance_grad` (negative inside); one type per shape, outside is `NotRegion` — `HollowSphere` removed (`Sphere & ~Sphere`; molpy's `__init__` re-export dropped in lockstep, 2026-09-14); boundaries closed (`Parallelepiped` was half-open); new `HalfSpace`, `Cylinder`, `Ellipsoid`, `Polyhedron` (watertight `TriMesh`), `SphereUnion` (atoms as a region, periodic minimum image); Python `TriMesh`, `io.read_stl`, `distance` on every region class, composed-`Region` pickling as an object tree (was a JSON recipe)
- `molrs_ffi::RegionRef` + capsule `molrs.RegionRef/<abi_line>` (`_ffi_regionref_capsule()` on every region class); `_ffi_abi_token()` is a 5-tuple (consumers read indices 0–1)
- prmtop-derived force fields declare `lj/cut` + `coul/cut` with explicit `coulomb`/`dielectric`/`cutoff` (`AMBER_COULOMB = 18.2223²`). A LAMMPS include written **with** its header changes from `pair_style lj/cut/coul/long 10 10` to `pair_style lj/cut/coul/cut 9 10`; `pair_coeff` lines are unchanged. NBFIX / 12-6-4 / multi-term-improper / non-uniform-SCEE prmtops are refused.


## v0.14.0 (releasing — 2026-09-20)

Prepared on molrs `chore/test-orthogonalization` (over `dev`) and molpy
`ci/precommit-uv-parity`. Gates re-measured at the tagged tree, not carried
over from an earlier run:

| Gate | Command | Result |
|---|---|---|
| molrs lib | `cargo test -p molcrafts-molrs --lib --features full,filesystem` | 2045 passed |
| molrs doctests | `cargo test -p molcrafts-molrs --doc --features full,filesystem` | 74 passed, 13 ignored |
| molrs-python | `tox -e py` in `molrs-python` | 557 passed |
| molpy | `pytest tests/ -n auto` | 1144 passed, 1 skipped |
| molpy lint | `tox -e lint` | green |

The earlier draft of this section recorded 2068 lib tests and 1188 molpy tests.
Both were stale: molpy lost the `pack` suite when `molpy.pack` was deleted, and
the molrs figure predates the test-orthogonalization deletions. Re-measure at
tag time rather than trusting a recorded number.

Branch topology checked before tagging: `origin/master` (v0.13.2) is an
ancestor of `dev`, so `dev` → `master` is a fast-forward and no published
0.13.x work is lost. The August drift note (dev 22 commits behind master) no
longer holds.

1. molrs — merge the branch into `dev`, then `dev` → `master`; tag
   `v0.14.0`; push the tag (the publish workflow does crates.io, npm, PyPI).
   The publish jobs need the `ci` job, so a red pipeline cannot publish.
2. molnex — replace the `.dev1` wheel under `.wheels-gh200` with the released
   0.14.0 wheel and run the chain import in the aarch64 venv.
3. molpy — no rebase needed (`upstream/master` is already an ancestor of
   `ci/precommit-uv-parity`); the pin is already
   `molcrafts-molrs>=0.14.0,<0.15`. **Drop** the `[tool.uv.sources]` path
   override — it is why CI cannot resolve molrs on a runner, where the sibling
   checkout does not exist — then `uv sync --extra dev` against the published
   wheel, `pytest tests/ -n auto`, `tox -e lint`, tag `v0.14.0` **after** the
   molrs tag, push.
4. molpy — `.pre-commit/check_molrs_pin_on_pypi.py` self-skips while the path
   source is active; it starts gating once the source is dropped and the wheel
   is on PyPI.

## v0.15.0 (releasing — 2026-09-21)

Prepared on molrs `feat/cgsmiles` (over `origin/master`, v0.14). The chain
`cgsmiles-01a` … `cgsmiles-02d` landed as five link commits plus one batch
commit; this section records what shipped, not what was noticed (the deferred
items live under the `cgsmiles-*` topics in `notes.md`).

- CGsmiles reader: `io::smiles::parse_cgsmiles` → `CGSmilesIR` (`levels`,
  `fragments`, `pairs`, resolved at parse time), `CGSmilesIR::to_atomistic`
  (per-atom `frag_id`) and `CGSmilesIR::to_fragment` (one template per
  atomistic definition); Python `molrs.io.CGSmilesIR` with the seven frozen
  record classes (`CGGraph`, `CGNode`, `CGEdge`, `CGFragmentDef`,
  `ResolvedPair`, `PairEnd`, `BondingDescriptor`). No `CGSmilesReader` in
  molrs: the reader-shaped API is molpy's.
- Fragment SMILES dialect: `parse_fragment_smiles`, `fragment_to_atomistic`,
  `write_fragment_smiles`; `BondingDescriptor` / `DescriptorKind` on the AST.
- `core::Fragment` — the third `MolGraph` leaf — with `Port`, `PortId`,
  `PortKind` (`$ < > !`), per-atom `frag_id`, `inherit_frag_ids`, Frame round
  trip; Python `molrs.Fragment`, `views.Port` (`anchor` / `handle_atom`),
  `views.Fragment` (`def_bond` / `def_port` through the native writers).
- `conformer::ElementGraph`; `Conformer::generate<M: ElementGraph>` returns the
  leaf type it was given (`Atomistic` or `Fragment`); Python
  `Conformer.generate` accepts either.
- `Atomistic` / `CoarseGrain::try_from_molgraph` resolve their standard kinds
  by name with an arity check (`MolGraph::try_register_kind`) instead of the
  dense-id fallback that aliased a foreign first-registered kind onto `bonds`.
- **Breaking (Rust):** `SmilesError::new` takes a fourth `notation` argument
  and `SmilesError` carries `.notation`; `SmilesErrorKind` gains the
  descriptor and `Cg*` variants (exhaustive matches must be extended);
  `core::system::mapping` (`CGMapping`, `WeightScheme`) is deleted — zero
  consumers in molrs or any sibling repo.
- **Breaking (Rust):** `MolGraph::add_node_with` returns
  `Result<NodeId, MolRsError>` instead of `NodeId`. A payload whose value
  contradicts the element type an existing node column holds for that key is
  data (`merge` and the leaves' `from_frame` hand over foreign bags), so the
  conflict is returned rather than asserted. The leaf constructors
  (`Atomistic::add_atom{,_xyz,_bare}`, `Fragment::add_atom_{xyz,bare}`,
  `CoarseGrain::add_bead{,_bare}`) keep their infallible signatures and now
  carry the `# Panics` contract that used to live on `add_node_with`.
- **Breaking (Rust / Python / WASM / C):** `to_frame` returns
  `Result<Frame, MolRsError>` — `MolGraph::to_frame` and the three leaf
  delegates `Atomistic::to_frame`, `CoarseGrain::to_frame`,
  `Fragment::to_frame`. It was a live panic, not style debt: the graph holds
  no dtype opinion about a key it has no column for, so
  `set_atom(id, "x", "left")` stores a str column under a key the Frame schema
  declares float and the emit-side refusal killed the process. Python
  `Atomistic.to_frame` / `CoarseGrain.to_frame` / `Fragment.to_frame` now
  raise `ValueError`; wasm `toFrame` and the perceive / conformer / typify
  entry points throw; `molrs_frame_from_smiles` can return
  `MolrsStatus::InternalError`.
- **Breaking (Python stubs only):** 17 stub declarations that mirrored no
  `_lib` class (`Parameters`, `Type`, the `*Type` / `*Style` view classes,
  `ChargeModel`, `Compute`) are removed from `_lib.pyi`;
  `tests/test_stub_parity.py` now guards class-name parity.
- **Breaking (Rust / Python / C) — force-field construction (system-forcefield-01):**
  a force field is built only through `ForceField::def_style(category, name, Params)
  -> Result<&mut Style, DefError>`, `Style::def_type(name, Params)` and
  `Style::def_type_at(name, endpoints, Params)` (endpoints given explicitly, for names
  outside the `-` grammar such as MMFF's `0_1_5`). Removed: `def_atomstyle` …
  `def_pairstyle`, `with_*style`, `def_atomtype` … `def_pairtype`, the old
  `ForceField::def_type(category, style, …)`, the panicking `Style::def_type`,
  `try_def_type`, `styles_mut`. `DefTypeError` is renamed `DefError` (new variant
  `UnknownStyle`). `Style.name` / `.params` / `.defs` are private — read them through
  `name()` / `params()` / `defs()`. Python mirrors the three primitives:
  `ForceField.def_style(category, name, params=None) -> Style`,
  `Style.def_type(name, params=None)`, `Style.def_type_at(name, endpoints,
  params=None)`; removed `def_*style`, the `def_style(Style)` overload, the unbound
  `Style()` form and `BondHarmonicStyle` / `AngleHarmonicStyle` / `DihedralOPLSStyle` /
  `PairCoulLongStyle`, and the per-category `def_type(itom, jtom, …)`. C:
  `molrs_ff_def_style(ff, category, …)` replaces `molrs_ff_def_{atom,bond,angle,pair}style`;
  `molrs_ff_def_type` no longer creates a missing style (InvalidArgument) and never
  panics on a malformed name; new `molrs_ff_def_type_at`. The C JSON round trip now
  carries string params, endpoints, every style's params and `special_bonds`, and
  refuses anything it cannot carry.
- **Fixed (Python):** force fields read in Python (`read_opls_xml`,
  `read_lammps_forcefield`) kept only numeric style params, so a declared
  `mixing` (combining rule) was dropped and σ combined with the kernel default.
  Energies from those readers change to the declared rule.
- **Breaking (Rust / Python / C) — one conflict rule (system-forcefield-02):** a
  type is identified by (category, style, name); re-defining it with the same
  endpoints and exactly equal params is a no-op, anything else is the new
  `DefError::TypeConflict`; re-defining a style with different style params is
  `DefError::StyleConflict` (previously ignored). Definitions no longer append
  duplicates. `Params` and the `*Type` structs derive `PartialEq`.
  `Style::rename_type` returns `Result<bool, DefError>` (renaming onto an existing
  name with different params is a conflict). `OPLSAATypifier::oplsaa()` returns
  `Self`. Python and C map conflicts to `ValueError` / `InvalidArgument`.
- **Breaking (readers) — system-forcefield-02:** prmtop bond / angle types no longer
  carry the table-row `id` param, a second parameter set under one name is an error
  instead of being silently dropped, and a missing `ATOM_TYPE_INDEX` entry is an
  error. GROMACS atom types carry only `[ atomtypes ]` parameters (molrs units:
  σ Å, ε kcal/mol, mass, charge, `ptype`, optional `bond_type` / `atomic_number`);
  the per-atom `[ atoms ]` strings (`nr`, `resnr`, `residu`, `atom`, `cgnr`,
  `charge`, `mass`, `typeB`, …) are no longer written onto atom types — read them
  with `io::data::top::read_top`. `[ atomtypes ]` without `[ defaults ]` is an
  error. The GROMACS writer now writes `[ atomtypes ]` and refuses a partially
  parameterised atom type. The OPLS XML reader refuses differing
  `<NonbondedForce>` charges for one type. Files these rules now refuse: a `.top`
  whose bonded rows give different per-instance parameters under one type tuple,
  and a prmtop with hydrogen mass repartitioning (one type, several masses).
- **Breaking (Rust / Python / C) — merge and declared state (system-forcefield-03):**
  new `ForceField::merge(&mut self, &ForceField) -> Result<(), DefError>`, the union
  replayed through the def primitives, all-or-nothing, carrying style params,
  `special_bonds` and `units` (new `DefError::UnitsConflict` /
  `SpecialBondsConflict`; `DefError` is `PartialEq` but no longer `Eq`). `units` moves
  into Rust (`set_units`, `units()` defaulting to `"real"`); `declared_units()` /
  `declared_special_bonds()` return `None` until declared. Removed:
  `ForceField::subset(&Frame)` (Rust and Python) — no replacement, `ff::forcefield`
  no longer depends on `Frame`; Python `ForceField.map_type` — the LAMMPS data writer
  assigns type ids; the Python-only `units` attribute — now the Rust property, and
  `ForceField(name, units=None)` declares units only when given (was `"real"`). The
  Python `merge` no longer drops `special_bonds` and style params, and a conflicting
  merge raises `ValueError` instead of silently keeping the first definition. The
  LAMMPS force-field reader declares `"lj"` for a `units lj` include (was read as
  `"real"`) and `"real"` otherwise; prmtop declares `"real"`. C JSON: `units` is
  carried, `special_bonds` is written only when declared, a non-string `units` is
  InvalidArgument.
- **Breaking (Rust / Python / wasm) — PotentialCompiler (system-forcefield-04):**
  compiling moves off the force field. Removed: `ForceField::to_potentials` /
  `to_typed_potentials` and `Style::to_potential` / `to_typed_potential` (Rust), and
  `ForceField.to_potentials` / `to_typed_potentials` (Python). Replacement:
  `ff::potential::PotentialCompiler::new(&ff).compile(&frame)` /
  `.compile_typed(&frame)`; Python `molrs.ff.PotentialCompiler(ff)` with
  `compile(frame)` (`None` is now a `TypeError`), `defer()` (the former
  `to_potentials(None)`, binding topology at evaluation) and `compile_typed(frame)`;
  the compiler holds a copy of the force field taken at construction.
  `molrs.md.MD.set_forcefield` now requires a `ForceField`. wasm keeps
  `typifier.toPotentials(frame)`; its error prefix changes to `toPotentials:`.
  Energies, forces and ETKDG coordinates are bitwise unchanged.
- **Fixed (Rust / Python) — OPLS strict typing:** strict mode (the default) returned
  `Ok` with atoms no def matched left untyped and their bonded terms skipped, so
  compiling failed later. It now returns an error naming every untyped atom.
- **Breaking / Fixed — OPLS-AA follows GROMACS (opls-gromacs 01–03):**
  - GROMACS force-field I/O is directive-only: the reader reads `[ defaults ]`
    (comb-rule 2/3 → `lj/cut` `mixing`), `[ atomtypes ]` (σ/ε on the `lj/cut` self row),
    `[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]` with the correct function
    codes (3 = Ryckaert–Bellemans; 2 and 3 were swapped), and `#include` / `#define` /
    `#ifdef`; molecule sections and unmodelled sections are refused by name unless
    skipped (`with_skipped_directive`; Python `skip_directives=`). The writer emits the
    same directives (OPLS torsions as code 3; it wrote them as zeros) and refuses what
    GROMACS cannot express. Removed: Rust `read_gromacs_top_ff`,
    `write_gromacs_top_ff(_str)` (use `GromacsTopFfReader` / `GromacsTopFfWriter`), the
    reader's public fields and `include_dirs`.
  - The OPLS-AA table is regenerated from GROMACS v2026.3 `oplsaa.ff` by `cargo
    mrs-gen-opls --gromacs <dir>`: classes are GROMACS `bond_type` (651 types used to
    carry their own name, leaving most bonded rows unreachable), the library declares
    geometric mixing (the kernel used arithmetic), and LAMMPS exports now state the
    kernel's rule (`pair_modify mix …`) when a style declares none. OPLS energies change.
  - OPLS typing rules are molrs-owned Daylight SMARTS (`ff::params::OPLSAA_TYPING`,
    `OplsRuleRow`), matched after aromaticity is perceived on a private copy; ranking is
    pairwise override dominance (`priorities()` / `LAYER_PRIORITY_STRIDE` removed); new
    diene types opls_150 / opls_178. Fixed: C=C / C=O molecules were untypeable,
    chlorobenzene's Cl was typed as chloride (net −0.82), pyridine / pyrimidine / pyrrole
    were untypeable, `[Cl,C,H]` read H as a count, a later level overwrote a type that
    overrides it. `from_xml_str` refuses override cycles and unknown overrides.
- **Breaking (Rust) / Fixed — one type-label grammar (system-forcefield-05):** new
  `core::store::type_labels::{TypeName, TypeLabels, BlockTypes}`. `TypeName` owns the
  endpoint grammar (`-`, or `::` when a part contains `-`; empty wildcard positions are
  kept) and an optional `@` qualifier (`with_qualifier` / `qualifier`), reordered with
  the endpoints by `reversed` / `canonical`. `TypeLabels::from_frame` is the Frame's
  type-id contract, used by the LAMMPS data writer; new `keys::*_TYPE_LABELS` meta
  keys. New `DefError::Name` for a malformed type name; `def_type` keeps an `@`
  qualifier in the name instead of folding it into the last endpoint. Stricter: a
  malformed type-label inventory meta value (`"1C"`, `"x:C"`, `"1:"`, a repeated id,
  a non-string value) is an error instead of being skipped. Fixed: an OPLS wildcard
  dihedral such as `-CA-CA-` was aliased to the bond label `CA-CA`, so
  `lammps_type_ids_from_frame` gave the bond the dihedral's id and
  `write_data_coeffs` wrote that bond's coefficients under the wrong type (seen for
  benzene with an inventory); `*.ff` includes now spell wildcard dihedrals with their
  empty position (`CA-CA-N-` → `-N-CA-CA`). Pure-label frames write byte-identical
  data files.
- **Breaking (Rust / Python) — label-driven LAMMPS coefficients (system-forcefield-06):**
  `LammpsFfWriter::new(&TypeLabels)` / `with_options(&TypeLabels, LammpsWriteOptions)`
  write the coefficients the system's labels need, looked up by name (bond / angle /
  dihedral labels in either orientation, impropers exactly). `LammpsWriteOptions`
  loses `atom_types` … `improper_types` and `type_ids`; `lammps_type_ids_from_frame`
  is deleted (Rust, Python). Python: `write_lammps_forcefield(path, forcefield, frame,
  *, precision, skip_pair_style, skip_units, units)`,
  `write_lammps_forcefield_str(forcefield, frame, …)`,
  `write_lammps_data_coeffs(forcefield, frame, *, precision, units)`. Force-field types
  no label uses are not written, and an unsupported style that holds only unused types
  is no longer an error. Refused now instead of written incomplete: a label with no
  type in the force field (including an atom label with no pair type, and an integer
  type id the force field lacks), and, in the data-file Pair Coeffs, an explicit cross
  pair between used types (use the `*.ff` include). Coefficients for OPLS and GAFF
  systems are byte-identical (`*.ff` lines may be reordered within a section).
- **Breaking (Rust) — typing is a template method (system-forcefield-07):** the
  `Typifier` trait now requires exactly `r#match(&self, &mut Atomistic) ->
  Result<Match, String>` and `library(&self) -> &ForceField` (no `type Mol`, no
  `typify`). New `Match`, `Annotation`, `Match::write_onto` and `Typing<T>`
  (`Typing::new(t).typify(&mol)` returns the typed copy; `forcefield()` is the output:
  exactly the types assigned so far). New `ForceField::empty_like`. Removed: the
  inherent `typify` / `ff` of `OPLSAATypifier`, `MMFF94Typifier`, `MMFF94STypifier` and
  `UFFTypifier`, and `LayeredTypingEngine::typify` (now `assign`). Replacement:
  `Typing::new(X::new()).typify(&mol)` and `.library()` / `.forcefield()`. `AtdTypifier`
  and `BCCAtomChargeTypifier` implement the trait with empty libraries. MMFF
  `stbn_type` is `{sbt}_{i}_{j}_{k}` in the angle's own node order; UFF bonded labels
  follow `TypeName` with `@` bond orders; OPLS atom types now also stamp `mass`, and
  estimator-filled terms get a `type`. UFF bond/angle parameters are computed on the
  canonical orientation (≤1 ulp change for reversed terms).
- **Fixed — MMFF / OPLS naming (system-forcefield-07):** an MMFF torsion found on the
  secondary-type restart or by the empirical rules is named `{tt}@{sec}_…` /
  `{tt}@{b}_…` (bond class `-` `=` `:` `~`), so ring torsions sharing `{tt}_{types}`
  with different parameters no longer collide. The OPLS/parmchk2 dihedral estimate is
  now the same whichever end it is read from (the penalty used to depend on atom order).
- **Fixed — OPLS estimator element classes:** all-caps OPLS classes (`CA CM CN CO CR CS
  CU NA NB NO OS HO HS`) were read as the elements Ca, Cm, …, Hs, so the estimator
  refused every element-compatible substitution at those slots (worse analogs, empirical
  fallbacks, dihedrals dropping to the `no_torsion` placeholder). A class now takes the
  element of its member types, and the token fallback only accepts true title case.
- **Breaking (Rust) — GAFF is a typifier (system-forcefield-08):** `gaff_forcefield(set,
  &Atomistic)`, `GaffError`, `MissingTerm` and `ff::forcefield::gaff` are removed.
  Replacement: `Typing::new(ff::typifier::GaffTypifier::new(set)).typify(&typed)` after
  ATD (`Typing::new(AtdTypifier::new(AtdParameterSet::Gff))`); `GaffParameterSet` stays
  at `ff::`. The library declares the AMBER 1-4 weights (lj 1/2, coul 1/1.2), so they
  reach the output like every other typifier's; typed atoms also get `mass`.
- **Breaking (Python / wasm) — typifier bindings (system-forcefield-09):** Python
  `molrs.ff.typifier.Typifier` is the one base: subclasses implement `match(graph) ->
  Match` (new `molrs.ff.typifier.Match(nodes, links=None, *, styles=(), pairs=())`) and
  may override `library()`; `typify` is final (defining it in a subclass is a
  `TypeError`) and returns a typed copy; `forcefield()` is now the output (the types
  assigned so far), returned as a copy — it used to return the library. Native
  `MMFF94Typifier`, `MMFF94STypifier`, `OPLSAATypifier`, `AtdTypifier` can no longer be
  subclassed. wasm typifiers wrap `Typing`; `toPotentials` compiles what `typify` wrote.
  Fixed: Python integer props outside the 32-bit range were silently wrapped, and past 64
  bits turned into floats; both now raise `OverflowError`.
- Re-deferred: the wasm `NeighborQuery` symmetry gate (`notes.md` § Known
  asymmetries, promised "to 0.15" on 2026-08-25) does not ship in 0.15.0 —
  wasm still has no consumer (facade-first), and deletion stays ruled out
  because `compute/hbond/detect.rs`, `compute/rdf/mod.rs`,
  `compute/dynamics/van_hove.rs` and `ff/potential/soft.rs` consume the engine
  type. Carried to the next minor.

Gates measured at the tree to be tagged (`docs/releasing.md:9-64`):

| Gate | Command | Result |
|---|---|---|
| molrs lib + doctests | `cargo test -p molcrafts-molrs --features full,filesystem,stream` | 2430 lib passed; 80 doctests passed, 11 ignored |
| molrs-ffi / molrs-cxxapi | `cargo test --manifest-path …` | 17 + 1 passed; 11 passed |
| fmt × 6, clippy × 5, rustdoc `-D warnings` | per `releasing.md` | all green |
| `cargo package` | `--manifest-path molrs/Cargo.toml` | packaged; `--list` inspected |
| molrs-python | `tox -e py` | 616 passed |
| molrs-wasm | `wasm-pack build` + `wasm-pack test --node` | build ok; 39 node tests passed |
| molrs-capi | `cmake` + `ctest` | configured, built; 1/1 ctest passed |

1. molrs — open the PR `feat/cgsmiles` → `master`, merge green, tag
   `v0.15.0` (must equal `Cargo.toml:12`), push the tag; Publish does
   crates.io, npm, PyPI (incl. Pyodide) and the C API assets.
2. molpy — after Publish, move `pyproject.toml:34` to
   `molcrafts-molrs>=0.15.0,<0.16` in the `backmap-` chain.
3. molpack — `molpack/Cargo.toml:26`, `molpack/python/Cargo.toml:22,28` pin
   `^0.14` on **path** dependencies into this checkout and hard-fail from the
   bump commit onward; move them to `^0.15` and rebuild against the 0.15
   `_ffi_abi_token` handshake line.
