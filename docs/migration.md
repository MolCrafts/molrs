# Migration guide

This guide lists the breaking changes in each minor release, grouped by
surface. Every downstream project pins one molrs minor line (see
[interop.md](interop.md)), so moving to a new minor means working through its
section. Changes that only add features are not listed here, apart from a
short "Also new" list at the end of each section.

## 0.14 → 0.15

### All surfaces

- **ABI line.** The capsule names are now `molrs.FrameRef/0.15`,
  `molrs.ForceFieldRef/0.15` and `molrs.RegionRef/0.15`, so handles cannot be
  exchanged with a 0.14 build. Rebuild consumer extensions and re-pin them to
  `>=0.15.0,<0.16`.
- **Frame vocabulary version is 2** (it was 1). This is
  `schema::FRAME_VOCAB_VERSION`, `molrs.schema.VOCAB_VERSION`,
  `schemaVocabVersion()` and `molrs_schema_vocab_version()`.
- **Floats are `f64` only.** f16/f32 columns and f32 meta values are gone.
  Zarr f16/f32 arrays and `"f32"` meta tags are refused on read, not widened.
  Python widens a float16/float32 array to float64 when it is inserted.
- **Image flags are `i32`** everywhere (SimBox, MD, binders). They were `i64`.
- **Frame meta keeps insertion order.** It used to be sorted by key, so every
  meta iterator and index-based accessor now returns keys in insertion order.
- **CoarseGrain frame layout.** A CoarseGrain frame now uses
  `atoms` + `bonds(atomi, atomj)`, plus an optional `members(ibead, atom)`
  block. 0.14 wrote `beads` + `cgbonds(ibead, jbead)`, and old frames in that
  shape no longer load. `from_frame` requires a `bead_type`, `type` or
  `type_id` column and refuses more than one `mol_id`.
- **Force-field model.** A force field is built only through
  `def_style` / `def_type`. Re-defining a style or type with different params
  is an error; an identical re-definition does nothing. Compiling is done by
  `PotentialCompiler`, not by the force field, and typing is done by
  `Typing<T>` (Rust), whose `forcefield()` holds only the types it assigned.
- **Energies change.**
  - The OPLS-AA tables were regenerated from GROMACS `oplsaa.ff` v2026.3: classes
    are now GROMACS `bond_type`, and mixing is geometric (it was arithmetic).
  - Python force fields from `read_opls_xml` / `read_lammps_forcefield` now
    honour the declared `mixing`.
- **GROMACS force-field I/O.**
  - Function codes 2 and 3 were swapped; now 3 = Ryckaert–Bellemans.
  - The reader refuses molecule sections and sections it does not model,
    unless you skip them.
  - `[ atomtypes ]` without `[ defaults ]` is an error.
  - Atom types carry only `[ atomtypes ]` params. Per-atom `[ atoms ]` data is
    read with `read_top`.
- **Stricter readers.**
  - prmtop bond/angle types no longer carry `id`.
  - A second parameter set under one type name is an error.
  - LAMMPS data refuses unknown sections and unparseable header lines unless you
    skip them. Header `units` are read into meta.
  - The LAMMPS ff reader declares `"lj"` for a `units lj` include.
  - Malformed type-label inventory meta (`"1C"`, `"x:C"`, repeated id) is an
    error.
- **LAMMPS coefficient output is driven by the frame's type labels.** Types
  no label uses are not written. A label with no matching type is refused, and
  so is an explicit cross pair in data-file `Pair Coeffs` (write it in the
  `*.ff` include instead).
- **`to_frame` can fail.** A node property whose dtype contradicts the
  schema, for example `set_atom(i, "x", "left")`, used to abort the process.
  It is now an error on every surface.
- **`*.mrec` records follow the molrec contract.**
  - `meta.molrec_version` is checked only when it is present. 0.14 refused a
    non-empty `meta` without it; 0.15 opens such a store. A present value must
    still be an integer in `1..=MOLREC_VERSION`, and `null`, `0`, a string, a
    float or a newer version is refused. Writers still stamp
    `molrec_version: 1` on every record.
  - An undefined cell (`box` with `cell_defined: false`) ignores its
    `vectors`. Readers accept any matrix there, zeros included, and do not
    invert it; writers write the identity. 0.14 refused a singular matrix
    even for an undefined cell.
  - A canonical identifier column (`id`, `atomic_number`, `mol_id`, `res_id`,
    `type_id`, `atomi`…`atoml`, `bond_type`, `bond_number`) stored at a width
    other than `u64` is refused on read, naming the column and its width. 0.14
    widened it silently. Inserting a narrow unsigned array in memory still
    widens it to `u64`. `SequenceSchema::declare_column` (Python
    `SequenceSchema.declare_column`) refuses such a key at a narrower width.
  - An observable whose `kind` is neither `scalar` nor `vector` no longer
    fails the record read. It is carried as `ObservableKind::Other(String)`
    and written back unchanged. Observable data with no
    `observables/meta/<name>` entry is still refused.
  - A root section the reader does not know is ignored. 0.14 read every
    unknown section as a frame group: arrays directly under it were dropped,
    and a sequence-shaped one could fail the read.
  - The frame, system and trajectory doors (`read_frame_file`,
    `read_system_file`, `read_trajectory_file`; Python `read_mrec`,
    `read_mrec_system`, `read_mrec_trajectory`) decode only `meta` and their
    own section. A broken trajectory or observables section no longer fails
    `read_mrec`. The lazy trajectory reader (`FrameSequence::open`, Python
    `TrajectoryReader`) and the WASM readers now validate
    `meta.molrec_version` too.
  - **Every reader decodes `zstd` and `numcodecs.shuffle`**, the wasm32
    build included (molrec's must-decode set). A build without
    `zarr-codecs` decodes `zstd` through a pure-Rust decoder (`ruzstd`), so
    the `zarr` feature now pulls `ruzstd` and `inventory`. It still cannot
    *encode* `zstd`. A store holding such an array — every `f64` column with
    a declared precision — cannot be read by molrs 0.14 or by any reader
    without the two codecs.
  - **Frame meta is typed on disk.** A `frame` / `system` group writes each
    meta value in its typed JSON form and adds the attribute `_meta_types`
    (`{key: tag}`), so a value reads back at its tag: an `i32` stays `i32`,
    an `f64x3` stays a vector, `1.0` stays `f64`. NaN and ±∞ are stored as
    `"NaN"` / `"Infinity"` / `"-Infinity"` (0.14 wrote JSON `null`, losing
    them), and a `u64`/`i64` beyond ±2⁵³ as a decimal string. A store without
    `_meta_types` is inferred as before. A meta key named `_meta_types` is
    refused at write, and a typed value in any other form is refused at
    read. The same forms are used for `sequence_schema` fills (a NaN fill is
    now storable; a `null` fill is refused) and for the `{dtype, value}`
    envelope of the `serde` / `stream` wire form.
  - **Declared precision.** An `f64` column may declare a precision `p`
    (`Block::set_precision`, Python `Block.set_precision`). Writers then
    store `round_half_even(x / q) · q` with `q` the largest power of two
    `≤ p` (error `≤ p/2`), pipe the column through `numcodecs.shuffle` +
    `zstd` level 3 (`gzip` level 1 without `zarr-codecs`), and record `p`:
    as the array attribute `precision` on `frame` / `system`, and in the
    column's `sequence_schema` entry on a trajectory. Readers hand back the
    stored values and the declaration (`Block::precision`). A trajectory
    compares the *rounded* values with the previous update, so a change
    below `q/2` writes no update; a frame stating a precision other than the
    pinned one is refused. Coordinates go from 24 to about 7.6 B/atom/frame
    at `p = 1e-3` Å. The default is unchanged: no precision, raw floats.

### Rust crate (`molcrafts-molrs`)

#### Cargo features

- **`default` is now `["rayon"]`**, so you get core only. It used to be
  `["full", "filesystem", "rayon"]`.
  ```toml
  molrs = { package = "molcrafts-molrs", version = "0.15", features = ["full", "filesystem"] }
  ```
- **Feature `ff` now enables `io`.**

#### Crate-root re-exports removed

| 0.14 | 0.15 |
|---|---|
| `molrs::{perceive_aromaticity, add_hydrogens, implicit_h_count, remove_hydrogens}` | `molrs::perceive::{aromaticity, hydrogens}::…` |
| `molrs::{RingInfo, find_rings, max_ring_system_size}` | `molrs::perceive::rings::…` |
| `molrs::{MatchOptions, Reaction, RingPrimitive, SmartsMatch, SmartsPattern}` | `molrs::perceive::smarts::…` |
| `molrs::{BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d, chiral_volume, find_chiral_centers}` | `molrs::perceive::stereo::…` |
| `molrs::compute_gasteiger_charges` | `molrs::ff::charge::compute_gasteiger_charges` |
| `molrs::smiles` | `molrs::io::smiles` |
| `molrs::Record` | `molrs::MolRec` |
| `molrs::SchemaValue` | `serde_json::Value` |

#### Store: column access

- **The typed getters are removed.** `Block::get_{float,int,uint,bool,u8,string}`
  and their `_mut` forms are gone. Use `Block::get(key) -> Option<&Column>` /
  `get_mut`, then a `Column::as_*` projection: `as_float`, `as_int` (i32),
  `as_uint` (u64), `as_bool`, `as_u8`, `as_string`, plus the new
  `as_i8/i16/i64/u16/u32/c64/c128`, each with a `_mut` variant.
  ```rust
  // 0.14
  let x = block.get_float("x").unwrap();
  let ids = block.get_uint_mut("id").unwrap();
  // 0.15
  let x = block.get("x").and_then(|c| c.as_float()).unwrap();
  let ids = block.get_mut("id").and_then(|c| c.as_uint_mut()).unwrap();
  ```
- **`BlockView::get_{float,…,string}` are removed.** Use
  `view.get(key) -> Option<&ColumnView>`, then `.as_float()` and so on, which
  return `ArrayViewD`.
- **Trait `ColumnAccess` is removed** (`as_*_view`, `nrows`, `dtype`,
  `shape`). Call the inherent `Column` / `ColumnView` methods instead.
- **`BlockAccess::get_*_view(key)` → `BlockAccess::column(key) -> Option<ColumnView>`.**
- **`FrameAccess::get_*(block, key)` → `FrameAccess::column(block, key)`.**
  ```rust
  frame.get_uint("bonds", "atomi")                          // 0.14
  frame.column("bonds", "atomi").and_then(|c| c.as_uint())  // 0.15
  ```
- **`Block::has_int` / `has_uint` cover every signed / unsigned width.**
  `as_int` / `as_uint` still return only i32 / u64 columns, so `has_int(k)`
  no longer guarantees that `as_int()` is `Some`. Match on `Column` or
  `dtype()` instead.
- **`Block::has_f32` is removed.**
- **`BlockError` has new variants** (`ValidityLength`, `MissingColumn`,
  `StackDtype`, `StackShape`), which breaks exhaustive `match`es.

#### Store: dtypes, meta, frame

- **f16/f32 removed.** `DType::{Float16, Float32}`, `Column::{Float16, Float32}`,
  `Column::{from_f16, from_f32, from_f16_holder, from_f32_holder, as_f16(_mut), as_f32(_mut)}`
  are gone. `BlockDtype` is no longer implemented for `f16` / `f32`, so call
  `mapv(f64::from)` before inserting.
- **f32 meta removed.** `MetaValue::{F32, F32x3, F32x6, F32x9}` and `as_f32`
  are gone; use `F64*` and `as_f64`. XTC/TRR `time` / `precision` meta are now
  `f64`.
- **`MetaMap` is an `IndexMap`.**
  - `iter()` returns `MetaIter`.
  - `impl IntoIterator for MetaMap` (by value) is removed; iterate `&meta`.
- **Ordered maps elsewhere.**
  - `Frame::into_inner` returns `IndexMap<String, Block>`.
  - `Frame::from_map` takes `impl IntoIterator<Item = (String, Block)>`.
  - `FrameView::from_parts` takes an `IndexMap`.
  - `Relation.props` is an `IndexMap`.
- **`store::frame::FRAME_SCHEMA_VERSION` is removed**, with no replacement.
- **`MolRec::extra_sections` is removed.** Unknown root sections are ignored
  on read, so nothing fills it. To read a producer's own frame-shaped section,
  call `io::mrec::read_frame_section_store(store, name)`.
- **`ObservableKind` gains `Other(String)`** and is no longer `Copy`.
  `ObservableKind::parse(s) -> Option<Self>` is replaced by
  `ObservableKind::from(s)`, which never fails, and `as_str` now returns a
  `&str` borrowed from the kind. Exhaustive `match`es must handle `Other`.

#### Schema and units

- **`keys::canonical_dtype(key)` → `schema::column(key).map(|s| s.dtype)`.**
- **`ColumnSpec.unit: &str` → `ColumnSpec.dimension: ColumnDim`.**
  `ColumnDoc` also gains a `dimension` field, which breaks struct literals.
- **`schema::Validator` is a unit struct.** `with_annotations`, `dtype_of` and
  `ViolationKind::AnnotationConflict` are removed.
- **`units::PRESET_DIMENSIONS` → `units::PresetDim::ALL`** (call `.name()` for
  the string).
- **`units::constants::PICOSECOND_S` is removed.**
- **The LJ preset temperature unit `lj_epsilon` is renamed `lj_epsilon_over_kB`.**

#### Math: `core::math` → `op`

- **`core::math::{norm3, cross3, mat3_mul_vec, det3, inv3, matmul}` are removed.**
  Use `molrs::op::vec3::{norm, cross, dot, …}` and
  `molrs::op::linalg::{det3, inv3}`. They take plain arrays
  (`Vec3 = [F; 3]`, `Mat3 = [[F; 3]; 3]`); convert from ndarray with
  `op::types::{to_vec3, to_mat3}`. For `matmul`, use ndarray `.dot()`.
- **`core::math::diagonalize` is removed** (`eigvals_sym_3x3`, `eigh_sym_3x3`,
  `eigh_largest_sym_4x4`). Use `op::linalg::{eigh_sym_3x3, eigh_sym_4x4}`; the
  largest 4×4 pair is the first one `eigh_sym_4x4` returns.
- **`ff::potential::geometry::{cross3, dot3, mag3}` → `op::vec3::{cross, dot, norm}`.**
- **`compute::density::kabsch` is removed** (`kabsch`, `det3`). Use
  `op::superpose::superpose(reference, target, weights, gap_tol)`.
- **`compute::environment::angular_separation::Quat` → `op::types::Quat`.**

#### Spatial

- **`geometry::rotate` returns `Result<(), MolRsError>`.** It refuses a zero
  or non-finite axis and a non-finite angle; it used to silently do nothing.
- **`geometry::align_direction` is removed.** Compose the replacement:
  ```rust
  if let Some((axis, angle)) = molrs::op::rigid::alignment(from_dir, to_dir) {
      geometry::rotate(mol, axis, angle, Some(from))?;
  }
  geometry::translate(mol, [to[0] - from[0], to[1] - from[1], to[2] - from[2]]);
  ```
- **`SimBox::try_new` is removed.** Use `SimBox::new` (same arguments) or
  `SimBox::from_matrix(h, origin, pbc)`.
- **`SimBox::new_cell(h, …, cell_defined: false)` ignores `h`.** The box
  carries the identity matrix, so `h_view()` / `matrix()` return the identity
  and a singular `h` is no longer an error. A defined cell is unchanged.
- **Image flags are `I` (i32), not `i64`.** This affects
  `SimBox::{wrap_shifts, images, unwrap}`, `periodic::ghosts::…::forward_comm`,
  `MDState.images`, and the `wrap_shifts` argument of `ForceProvider::compute` /
  `compute_into` and the `md::pairs` `advance` / `refresh` methods. Custom
  `ForceProvider` impls must update.

#### Molecular graphs (`core::system`)

- **`system::mapping` is removed**: `CGMapping`, `WeightScheme`,
  `coarsen_with_templates`, `backmap` and their root re-exports. The nearest
  replacements are `perceive::Coarsener` (coarsening),
  `CoarseGrain::from_atom_frame` and `builder::Assembler` (backmapping). They
  are not 1:1.
- **Graph methods now return `Result`.**
  - `MolGraph::add_node_with` returns `Result<NodeId, MolRsError>`.
  - `MolGraph::merge`, `Atomistic::merge` and `CoarseGrain::merge` return
    `Result<…, MolRsError>`. A property conflict is now an error.
  - `Atomistic::to_frame` and `CoarseGrain::to_frame` return
    `Result<Frame, MolRsError>`.
- **`from_frame` / `read_frame` refuse what they used to drop.** That covers a
  relation block missing an endpoint column, a relation prop whose dtype
  conflicts with the kind, and a registered relation block the receiver lacks.
  Null cells in nullable columns stay null; they used to become `0`/`""`/`false`.
- **`MolGraph::node_table_mut`, `molgraph::Group` and `GroupId` are removed.**
- **`Topology::from_frame` returns `TopologyError`.** It returned `MolRsError`;
  `TopologyError` converts with `From`, so `?` still works.
- **`Topology::add_angles` is removed.** Loop over `add_angle`.
- **`Element::is_early_atom` is removed.**

#### I/O

- **`io::reader::{open_txt, open_gz}` are removed.**
- **`io::zarr::schema::validate_trajectory(t)` is removed.** Call
  `t.validate()?` and `frame.validate()?` on each frame.
- **`io::data::lammps_data::lammps_type_ids_from_frame` →
  `core::store::type_labels::TypeLabels::from_frame(&frame)?`.**
- **PDB.** `io::data::pdb::{ModelRecord, parse_model_record, is_ter, HetAtmRecord}`
  are removed. `parse_hetatm_record` returns `Option<AtomRecord>`.
- **SMILES errors.** `SmilesError::new(kind, span, input)` →
  `SmilesError::new(kind, span, input, notation)`, and `SmilesError` gains a pub
  `notation` field. `SmilesErrorKind` gains descriptor and `Cg*` variants,
  which breaks exhaustive `match`es.
- **`smiles::AtomSpec` gains a pub `descriptors` field**, which breaks struct
  literals.
- **`LAMMPSDataReader` refuses unknown sections.** Opt out per section with
  `.with_skipped_section("Ellipsoids")`.
- **`stream::Publisher::send_async` → `send`.**
- **`MetaValue::to_attr_value` is removed.** Use
  `MetaValue::to_typed_json` (typed JSON form) and
  `MetaValue::from_typed_json(tag, value)`; `from_attr_value` stays, as the
  inference of an untyped value. `MetaValue::from_json_value` now reads the
  typed JSON payload forms (`"NaN"`, decimal strings beyond 2⁵³).
- **`io::mrec::write_trajectory_file(path, trajectory)` →
  `write_trajectory_file(path, trajectory, meta)`.** `meta` is an
  `Option<&JsonMap>`, like `write_frame_file` and `write_system_file`. Pass
  `None` to keep the old behaviour.

#### Force field: construction

- **Removed from `ForceField`:**
  `def_{atom,bond,angle,dihedral,improper,pair}style`,
  `with_{atom,bond,angle,pair}style`, `def_type(category, style, name, params)`,
  `styles_mut`, `remove_style`, `subset(&Frame)` (no replacement) and
  `to_potentials` / `to_typed_potentials` (see the next subsection).
- **Removed from `Style`:** `def_{atom,bond,angle,dihedral,improper,pair}type`,
  `try_def_type`, the panicking `def_type(name, &[(&str, f64)])` and
  `to_potential` / `to_typed_potential`.
- **The two remaining entry points** are
  `ForceField::def_style(category, name, Params) -> Result<&mut Style, DefError>`
  and `Style::def_type(name, endpoints, Params) -> Result<&mut Style, DefError>`.
  - Endpoints are explicit; the name is never split on `-`.
  - The number of endpoints follows the category: atom 0, pair 1 or 2, bond 2,
    angle 3, dihedral and improper 4.
  ```rust
  // 0.14
  ff.def_bondstyle("harmonic").def_bondtype("a", "b", &[("k", 100.0), ("r0", 1.2)]);
  ff.def_pairstyle("lj/cut", &[]).def_pairtype("a", None, &[("epsilon", 0.1), ("sigma", 3.0)]);
  // 0.15
  ff.def_style("bond", "harmonic", Params::new())?
      .def_type("a-b", &["a", "b"], Params::from_pairs(&[("k", 100.0), ("r0", 1.2)]))?;
  ff.def_style("pair", "lj/cut", Params::new())?
      .def_type("a", &["a"], Params::from_pairs(&[("epsilon", 0.1), ("sigma", 3.0)]))?;
  ```
- **`Style.name` / `.params` / `.defs` are private.** Read them with `name()` /
  `params()` / `defs()`; write params with `set_param` / `set_str_param`.
- **`Style::rename_type` returns `Result<bool, DefError>`** (it returned `usize`).
- **`DefTypeError` is renamed `DefError`.**
  - New variants: `UnknownStyle`, `TypeConflict`, `StyleConflict`,
    `UnitsConflict`, `SpecialBondsConflict`, `Name`.
  - It derives `PartialEq` but no longer `Eq`.
- **Units and special_bonds are declared state.** `units()` defaults to
  `"real"`; `declared_units()` / `declared_special_bonds()` return `None`
  until something declares them.

#### Force field: compiling

- **`ff.to_potentials(&frame)` → `PotentialCompiler::new(&ff).compile(&frame)`**,
  and `to_typed_potentials` → `compile_typed`
  (`molrs::ff::potential::PotentialCompiler`).
- **`ff::potential::extract_coords(&frame)` → `frame.coords()?`**, which
  returns N×3; flatten it if you need the old layout.
  **`write_coords` → `frame.set_coords(view)`.**
- **`Potentials::members_mut` is removed.**

#### Force field: typing

- **The `Typifier` trait now requires exactly two methods:**
  `r#match(&self, &mut Atomistic) -> Result<Match, String>` and
  `library(&self) -> &ForceField`. `type Mol` and `fn typify` are gone.
  `Typing<T>` runs a typifier and owns the output force field.
- **The inherent `typify()` / `ff()` methods are removed** from
  `MMFF94Typifier`, `MMFF94STypifier`, `OPLSAATypifier` and `UFFTypifier`.
  Wrap the typifier in `Typing` instead. `AtdTypifier` and
  `BCCAtomChargeTypifier` are wrapped the same way.
  ```rust
  // 0.14
  let t = MMFF94Typifier::new();
  let typed = t.typify(&mol)?;
  let pots = t.ff().to_potentials(&typed.to_frame())?;
  // 0.15
  let mut typing = Typing::new(MMFF94Typifier::new());
  let typed = typing.typify(&mol)?;
  let frame = typed.to_frame()?;
  let pots = PotentialCompiler::new(typing.forcefield()).compile(&frame)?;
  ```
  `typing.forcefield()` holds only the types assigned so far; use
  `typing.library()` to get the whole library.
- **GAFF is a typifier.** `ff::gaff_forcefield`, `ff::forcefield::gaff`,
  `GaffError` and `MissingTerm` are removed. `GaffParameterSet` still resolves
  at `ff::`.
  ```rust
  let (typed, ff) = gaff_forcefield(set, &atd_typed)?;                    // 0.14
  let mut g = Typing::new(GaffTypifier::new(set));                        // 0.15
  let typed = g.typify(&atd_typed)?;
  let ff = g.forcefield();
  ```
- **OPLS.**
  - `OPLSAATypifier::oplsaa()` returns `Self`; it returned `Result<Self, String>`.
  - `LayeredTypingEngine::typify` → `assign`.
  - Removed: the `opls::typing` module (`typify_atoms`),
    `opls::{typify_bonded, typify_bonded_with}`, `LAYER_PRIORITY_STRIDE` and
    `OplsTypingMeta::priorities()`. Ranking is now pairwise override dominance.
  - `ff::params::OplsAtomRow` loses `def`, `overrides`, `priority` and `layer`.
    The typing rules are now `OplsRuleRow` in `ff::params::OPLSAA_TYPING`.
    `OplsAtomRow.class` is now the GROMACS `bond_type`.
  - Strict typing errors on any untyped atom; it used to return `Ok` with the
    atom left untyped.
- **Label formats.**
  - MMFF `stbn_type` is `{sbt}_{i}_{j}_{k}`.
  - MMFF torsions found on restart are `{tt}@{sec}_…`.
  - UFF bonded labels use `TypeName` with `@` bond orders.
- **`ff::mmff::MmffMolProperties::is_setup_complete` is removed.**

#### Force field: readers and writers

- **GROMACS.**
  - `ff::read_gromacs_top_ff(path)` → `GromacsTopFfReader::new().read(path)`.
  - `ff::write_gromacs_top_ff(path, ff, p)` →
    `GromacsTopFfWriter::new().with_precision(p).write(ff, path)`;
    `write_gromacs_top_ff_str` → `.write_str(ff)`.
  - `GromacsTopFfReader` loses its public fields and `include_dirs`. Use
    `with_include`, and `with_skipped_directive(name)` to read past a section.
- **LAMMPS.**
  - `LammpsFfReader::with_default_units` is removed; units come from the file
    and default to `"real"`.
  - `LammpsFfWriter::new()` → `LammpsFfWriter::new(&labels)`, and
    `with_options(opts)` → `with_options(&labels, opts)`, where
    `labels = TypeLabels::from_frame(&frame)?`. The writer is now
    `LammpsFfWriter<'a>`.
  - `LammpsWriteOptions` loses `atom_types` … `improper_types` and `type_ids`.

#### Perception, compute, optimize, builder

- **Perception returns `Result`.** `perceive::hydrogens::{add_hydrogens, remove_hydrogens}`
  and `Perceive::find_hydrogens` return `Result<Atomistic, MolRsError>`.
- **Rotatable bonds take a policy.** `detect_rotatable_bonds(g)`,
  `detect_rotatable_bonds_with_downstream(g)` and `Perceive::find_rotatable(mol)`
  take a second argument, `UnknownBondPolicy`.
  `UnknownBondPolicy::NotRotatable` is the default.
- **`compute::util` helpers removed.** `{get_f_slice, get_positions, mic_disp, PositionSlices}`
  are gone; use `get_positions_ref` and `MicHelper::from_simbox(..).disp(..)`.
- **`VanHoveResult::self_second_moment` is removed.**
- **`optimize::SoftSpec::with_{sigma, repulsion, rcut, bond_k, angle_k}` are removed.**
  The defaults are now fixed.
- **`builder::WalkOutput.paths: Vec<Vec<F3>>` → `traces: Vec<Trace>`.** Read
  the points with `.points()`.
- **`conformer::distgeom::experimental_torsions_with_provenance` is removed.**
- **`ScaleLjError` gains `InvalidMass`**, which breaks exhaustive `match`es.

### Python (`molrs`)

#### Modules and top-level names

- **Removed modules:** `molrs.frame`, `molrs.views`, `molrs.ff.forcefield` and
  `molrs.ff.potential.soft`.
  - `Block`, `Frame`, `Atomistic`, `CoarseGrain`, the view classes and
    `NodeRef` / `RelationRef` / `Refs` / `RelationBuckets` are native classes
    at `molrs.<Name>`.
  - `ForceField`, `Style`, `Type` and the `*Style` / `*Type` classes are at
    `molrs.ff.<Name>`.
  - Pickles that name `molrs.frame.*` or `molrs.views.*` will not unpickle.
  - `SoftPotential` has no replacement.
- **Removed top-level names:** `molrs.FRAME_SCHEMA_VERSION` and `molrs.GraphViews`.
- **Rigid-body functions are now chainable methods** on `Atomistic` and
  `CoarseGrain`:
  - `molrs.translate(mol, d)` → `mol.translate(d)`
  - `molrs.rotate(mol, axis, angle)` → `mol.rotate(axis, angle, about=None)`
  - `molrs.scale(mol, s)` → `mol.scale([s, s, s], about=None)`
- **`molrs.align_direction` is removed.** Compose `rotate` and `translate`, or
  use `molrs.op.superpose`.
- **`molrs.schema.VOCAB_VERSION` is still exported**, but its value is now 2.
- **`ColumnSpec` changed.** It gains a `dimension` field before `unit`, which
  shifts positional construction. Schema `uint` keys are now `uint64` (they
  were `uint32`).

#### Block

- **`Block` is no longer a `MutableMapping`.**
  - Removed: `get`, `items`, `values`, `pop`, `update`, `setdefault`, `clear`.
  - Kept: `keys()`, iteration, `in` and `b[key]`.
- **Removed typed accessors.** `get_f32` / `get_f64` / `has_f32` are gone. Use
  `b[key]`, and `has_f64` / `has_int` / `has_uint` / `has_string` /
  `dtype(key)` to check the type.
- **Other removals:**

  | 0.14 | 0.15 |
  |---|---|
  | `b.view(key)` | `b[key]` |
  | `b.to_dict()` | `{k: b[k] for k in b}` |
  | `Block.from_dict(d)` | `Block(d)` |
  | `b.sort_(key)` (in place) | `b = b.sort(key)` |
  | `b.iterrows()`, `b.itertuples()` | removed |
  | `Block(vars_, nrows, shape)` | `Block(data=None)`, then `.resize(n)` / `.set_shape(s)` |
- **Indexing changed.** `b[i]` with an int now raises `TypeError` (it returned
  a row dict). `b[callable]` is removed.
- **`b.shape` changed.** It was `(nrows, ncols)`; it is now `[nrows]`, or the
  N-D grid shape.
- **Tuple-key write spreads columns.** `b["x", "y", "z"] = arr` now spreads an
  `(N, k)` array over k columns. It used to store one column named
  `"('x', 'y', 'z')"`.

#### Frame and meta

- **Removed typed accessors:** `has_f32` / `has_f64` / `get_f32` /
  `get_f64(block, key)`. Use `frame[block][key]`.
- **`to_dict()` and the `blocks` property are removed.** Use
  `[frame[k] for k in frame.keys()]`.
- **`Frame(blocks, meta, box)` takes `meta` and `box` as keyword-only
  arguments.** Passing an existing `Frame` to `Frame(...)` is removed; use
  `.copy()`.
- **JSON meta comes back frozen.** A JSON object in `frame.meta` is now a
  frozen `MetaDocument`, not a `dict`, and arrays come back as tuples.
  ```python
  frame.meta["run"]["step"] = 3            # 0.14; raises in 0.15
  doc = frame.meta["run"].copy(); doc["step"] = 3; frame.meta["run"] = doc
  ```
- **`Atomistic.to_frame` / `CoarseGrain.to_frame` raise `ValueError`** on a
  dtype conflict.

#### Graph views

- **`Atom.is_virtual` → `isinstance(atom, molrs.VirtualSite)`.**
- **`Angle` / `Dihedral` / `Improper` `itom` … `ltom` and `CGBond.ibead` /
  `jbead` → `.endpoints`.** `Bond.itom` / `jtom` remain.
- **`CoarseGrain.del_bead(b)` → `cg.despawn(b.handle)`.**
- **The `nodes` property is removed** (it came from `GraphViews`).
- **`RelationBuckets` keeps only `exact_bucket(cls)`.** Custom relation view
  classes can no longer be registered.
- **`Refs` is no longer a `list` subclass.** Use `list(refs)`.
- **`NodeRef` loses `setdefault` / `pop`.**
- **`NodeRef`, `RelationRef` and `Refs` cannot be constructed directly.**
- **Constructors take fewer arguments.** `Atomistic(...)` /
  `CoarseGrain(...)` take only `**props`, and `Graph()` takes no arguments.
- **`replicate` grows `self` in place:**
  ```python
  w = mol.replicate(n)                                     # 0.14
  w = molrs.Atomistic()                                    # 0.15
  w.replicate(mol, np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)),
              np.arange(n, dtype=np.int32))
  ```
- **Integer props outside the int32 range raise `OverflowError`.** They used
  to wrap silently.

#### Box, MD, units

- **`Box.matrix` → `Box.h`.**
- **Image flags are int32.** This applies to `Box.images`,
  `Box.unwrap(images=)` and `MD` `images`. int64 input is refused; use
  `.astype(np.int32)`.
- **`UnitRegistry.Unit(expr)` / `.Quantity(v, expr)` → `.parse(expr)` /
  `.quantity(v, expr)`.**
- **`molrs.md.MD.set_forcefield` requires a `ForceField`.**

#### `molrs.io`

- **Trajectory functions renamed:**

  | 0.14 | 0.15 |
  |---|---|
  | `read_trr`, `read_xtc` (eager list) | `read_trr_trajectory`, `read_xtc_trajectory` (lazy; `list(...)` for a list) |
  | `write_dcd`, `write_trr`, `write_xtc`, `write_lammps_traj` | `write_{dcd,trr,xtc,lammps}_trajectory` |
  | `read_gro` → `list[Frame]` | `read_gro` → first `Frame`; `read_gro_trajectory` for all |
  | `lammps_type_ids_from_frame` | removed; the force-field writers take the frame |
- **Removed parameters.** The reserved `frame=` parameter is gone from
  `read_xyz`, `read_pdb`, `read_mol2`, `read_lammps_data`,
  `read_lammps_molecule`, `read_lammps_trajectory`, `read_amber_inpcrd`,
  `read_amber_prmtop`, `read_top` and `read_xsf`. `atom_style=` is gone from
  `read_lammps_data` / `write_lammps_data`.
- **`molrs.io.raw` renames:**
  - `read_lammps` / `write_lammps` → `read_lammps_data` / `write_lammps_data`
  - `read_lammps_traj` / `write_lammps_traj` → `*_lammps_trajectory`
  - `read_dcd` / `read_trr` / `read_xtc`, and the `write_*` forms → `*_trajectory`
  - `read_chgcar_file` / `read_cube_file` / `write_cube_file` →
    `read_chgcar` / `read_cube` / `write_cube`
- **`molrs.io.mrec` renames:**
  - `read_frame` / `write_frame` → `molrs.io.read_mrec` / `write_mrec`
  - `read_system` / `write_system` → `read_mrec_system` / `write_mrec_system`
  - `read_trajectory` / `write_trajectory` → `read_mrec_trajectory` /
    `write_mrec_trajectory`
  - `read_meta` → `read_mrec_meta`
  - `sections` → `mrec_sections`
- **`GroFieldFormatter` no longer maps `resid` / `atom_id`.**
- **`meta=` takes what `frame.meta` hands out.** `write_mrec`,
  `write_mrec_system`, `write_mrec_trajectory` (which gains `meta=`) and
  `TrajectoryWriter(meta=)` accept a `dict`, a `MetaDocument`, or any mapping
  (`frame.meta` included), with nested tuples and documents. 0.14 took only a
  `dict` of lists and dicts, and raised `TypeError` for anything else.
- **A non-finite float inside a JSON meta document raises `ValueError`.**
  `frame.meta["doc"] = {"t": float("nan")}` used to store `null`. A
  top-level `float("nan")` is an `f64` value and is kept.
  `SequenceSchema.declare_meta_with_fill` infers an untagged fill the way
  `frame.meta` does, so a NaN fill is an `f64` fill.
- **Trajectory `time` has no unit.** The `TrajectoryReader.time` and
  `TrajectoryWriter.append(time=)` docs no longer say fs. A record does not
  store a unit for time; the producer's convention applies.

#### `molrs.ff`

- **Styles are defined through one method.**
  `def_{atom,bond,angle,dihedral,improper,pair}style(name, …)` →
  `def_style(category, name, params=None)`. Also removed:
  `def_style(StyleInstance)`, the unbound `Style()`, `BondHarmonicStyle`,
  `AngleHarmonicStyle`, `DihedralOPLSStyle`, `PairCoulLongStyle` and
  `Parameters`.
- **`Style.def_type` takes the type name first,** and its endpoints must be
  `AtomType` handles; strings raise `TypeError`.
  ```python
  # 0.14
  ff.def_atomstyle("full").def_type("c"); ...
  ff.def_bondstyle("harmonic").def_type("c", "h", k=340.0, r0=1.09)
  ff.def_pairstyle("lj/cut", cutoff=10.0).def_type("c", epsilon=0.1, sigma=3.4)
  # 0.15
  ats = ff.def_style("atom", "full")
  c, h = ats.def_type("c"), ats.def_type("h")
  ff.def_style("bond", "harmonic").def_type("c-h", c, h, k=340.0, r0=1.09)
  ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type("c", c, epsilon=0.1, sigma=3.4)
  ```
- **Removed `ForceField` methods:** `rename_type`, `remove_type`,
  `remove_style`, `subset(frame)` and `map_type` (no replacements);
  `def_*type`, `def_type(category, …)`, `style_params`, `types`,
  `type_endpoints`, `set_type_param` and `set_type_str_param`.
- **Renamed or reshaped `ForceField` members:**
  - `style_names()` → `[(s.category, s.name) for s in ff.styles]`
  - `special_bonds_lj` / `special_bonds_coul` are removed; declare the
    triples with `set_special_bonds(lj, coul)`
- **`ForceField(name, units="real")` → `units=None`.** It declares no units
  unless given; the `units` getter still reports `"real"`.
- **`merge` carries more and fails loudly.** It now carries `special_bonds`,
  style params and units, and raises `ValueError` on a conflict instead of
  keeping the first definition.
- **`Type` changes.**
  - `Type.params` is a plain `dict` (no `.kwargs` / `.args`).
  - `Type[k]` returns `None` for a missing key.
  - `*Type.matches()` is removed.
- **Compiling moved to `PotentialCompiler`:**
  - `ff.to_potentials(frame)` → `molrs.ff.PotentialCompiler(ff).compile(frame)`
  - `ff.to_potentials(None)` → `PotentialCompiler(ff).defer()`
  - `ff.to_typed_potentials(frame)` → `PotentialCompiler(ff).compile_typed(frame)`
- **`extract_coords(frame)` → `frame.coords.ravel()`**, or pass the frame to
  `Potentials` directly.
- **The `*_str` readers and writers are removed**: `read_forcefield_xml_str`,
  `read_opls_xml_str`, `read_lammps_forcefield_str`, `read_amber_prmtop_ff_str`,
  `read_gromacs_top_ff_str`, `write_gromacs_top_ff_str` and
  `write_forcefield_xml_str`. Use the path-based functions, which accept
  `PathLike`. `write_lammps_forcefield_str` remains.
- **LAMMPS writers take the frame.** The new signatures are
  `write_lammps_forcefield(path, ff, frame, *, precision, skip_pair_style, skip_units, units)`,
  `write_lammps_forcefield_str(ff, frame, …)` and
  `write_lammps_data_coeffs(ff, frame, *, precision, units)`. `frame` is
  required, the options are keyword-only, and `atom_types` … `improper_types` /
  `type_ids` are removed.
- **Typifiers implement `match`.** A `molrs.ff.typifier.Typifier` subclass
  implements `match(graph) -> Match`. Defining `typify` in a subclass raises
  `TypeError`.
  - `typify` returns a typed copy.
  - `forcefield()` returns the output (the types assigned so far, as a copy);
    use `library()` for the full library.
  - The native typifiers cannot be subclassed.
  - `molrs.ff.typifier.TGraph` is removed.
  ```python
  t = molrs.ff.MMFF94Typifier()
  typed = t.typify(mol)
  pots = molrs.ff.PotentialCompiler(t.forcefield()).compile(typed.to_frame())
  ```

### WASM / JS (`@molcrafts/molrs`)

- **Block typed accessors are removed:**

  | 0.14 | 0.15 |
  |---|---|
  | `hasF32`, `hasF64`, `hasI32`, `hasU32`, `hasStr` | `has(key)` + `dtype(key)` |
  | `getF32`, `getF64`, `getI32`, `getU32`, `getStr` | `get(key, fallback?)` |
  | `copyColF`, `copyColI32`, `copyColU32`, `copyColStr` | `copy(key)` |
  | `viewColF`, `viewColI32`, `viewColU32` | `view(key)` |
  | `setColF`, `setColI32`, `setColU32`, `setColStr` | `set(key, data, shape?)` |
  | `createColF`, `createColI32`, `createColU32` | `set(key, new Float64Array(n), shape)` |
  - `set` infers the dtype from the typed array. It refuses `Float32Array`,
    plain `number[]` and mixed arrays.
  - `view` / `copy` return the column's own typed array; an i64 column comes
    back as a `BigInt64Array`.
  - `dtype(key)` throws for a missing key (it returned `undefined`).
- **`Block.nrows()` → `Block.nrows`** (a getter).
- **`Block.shape` changed.** The old no-argument `shape()` is now the getter
  `structuralShape` (`number[] | undefined`). `shape(key)` returns one
  column's shape, and `setShape` takes `number[]`.
- **Frame methods renamed:**

  | 0.14 | 0.15 |
  |---|---|
  | `getBlock(k)` → `undefined` if absent | `get(k)`, throws if absent; check with `has(k)` |
  | `insertBlock(k, b)` (moves `b`) | `set(k, b)` (deep copy; `b` stays usable) |
  | `removeBlock(k)` | `remove(k)` |
  | `blockNames()` | `keys()` (insertion order) |
  | `frame.renameColumn(block, old, new)` | `frame.get(block).renameColumn(old, new)` |
  | Frame-level `has*` / `get*` typed accessors | `frame.get(block).get(key)` |
- **`Mesh.verticesF32` / `faceNormalsF32` → `vertices` / `faceNormals`.**
- **`Wasm*Stream` classes changed.** `parseRangeInInput` returns a `Frame`
  that the caller must free.
  - Removed: `releaseFrame`, `blockCount`, `blockName`, `columnCount`,
    `columnName`, `columnDtype`, `columnLen`, `columnPtrF64`, `columnPtrU32`,
    `columnPtrI32`, `columnStrings`, `boxH`, `boxOrigin` and `boxPbc`.
  - Read the returned Frame with `keys()`, `get()`, `view()`, `copy()` and
    `.box` instead.
- **`schemaColumnDtype(key)` returns `"f64"` / `"i32"` / `"u64"`** (it returned
  `"float"` / `"int"` / `"uint"`). `schemaDocument()` keeps the old names.
- **Typifiers keep state.** `UFFTypifier`, `MMFF94Typifier` and
  `MMFF94STypifier` wrap a `Typing`.
  - `toPotentials` compiles only what `typify` assigned, so call `typify` first.
  - A conflicting re-typing throws.
- **Error prefixes changed:** `to_potentials:` → `toPotentials:` (typifiers) and
  `compile:` (`LBFGS.run`).
- **New throws.** SMILES → Frame, the conformer, perception, `typify`,
  `findHydrogens` and `removeHydrogens` can throw `"toFrame: …"` when a
  property contradicts the schema.

### C API (`molrs.h`)

- **`MolrsDType` reports the stored variant.** Values 0–4 keep their names and
  numbers, but each now means exactly one type; 5–12 are new.

  | Value | Constant | 0.15 meaning | 0.14 meaning |
  |---|---|---|---|
  | 0 | `MOLRS_D_TYPE_FLOAT` | f64 | any float width |
  | 1 | `MOLRS_D_TYPE_INT` | i32 | i8/i16/i32/i64 |
  | 2 | `MOLRS_D_TYPE_BOOL` | bool (1 byte) | bool |
  | 3 | `MOLRS_D_TYPE_U_INT` | u64 | u8/u16/u32/u64 |
  | 4 | `MOLRS_D_TYPE_STRING` | string | string |
  | 5 | `MOLRS_D_TYPE_INT8` | i8 | — |
  | 6 | `MOLRS_D_TYPE_INT16` | i16 | — |
  | 7 | `MOLRS_D_TYPE_INT64` | i64 | — |
  | 8 | `MOLRS_D_TYPE_U8` | u8 | — |
  | 9 | `MOLRS_D_TYPE_U_INT16` | u16 | — |
  | 10 | `MOLRS_D_TYPE_U_INT32` | u32 | — |
  | 11 | `MOLRS_D_TYPE_COMPLEX64` | 2×f32 | — |
  | 12 | `MOLRS_D_TYPE_COMPLEX128` | 2×f64 | — |

  An i64 column no longer reports `INT`. A `switch` on
  `molrs_block_col_dtype` must handle 5–12.
- **Three dtype-agnostic accessors replace the nine typed ones.**
  `molrs_block_get_{F,I,U}`, `molrs_block_get_{F,I,U}_mut` and
  `molrs_block_copy_{F,I,U}` are replaced by `molrs_block_get`,
  `molrs_block_get_mut` and `molrs_block_copy`. The dtype is an
  out-parameter.
  ```c
  /* 0.14 */
  const double* x; size_t n;
  molrs_block_get_F(blk, key, &x, &n);
  /* 0.15 */
  const uint8_t* raw; size_t n; MolrsDType dt;
  molrs_block_get(blk, key, &raw, &n, &dt);
  if (dt == MOLRS_D_TYPE_FLOAT) { const double* x = (const double*)raw; /* … */ }
  ```
  - `out_len` is still an element count, but `molrs_block_copy`'s buffer size
    is now in **bytes**.
  - A string column is `MOLRS_STATUS_TYPE_MISMATCH`; in 0.14 it was
    `KEY_NOT_FOUND`. A missing key is still `KEY_NOT_FOUND`.
  - `molrs_block_set_{F,I,U}` and `molrs_block_col_commit` are unchanged.
- **`molrs_ff_def_{atom,bond,angle,pair}style` →
  `molrs_ff_def_style(ff, category, name, keys, vals, n)`.** It covers every
  category, including dihedral and improper. Conflicting params return
  `INVALID_ARGUMENT`.
- **`molrs_ff_def_type` takes explicit endpoints** (9 arguments; it took 7).
  ```c
  molrs_ff_def_type(ff, "bond", "harmonic", "CT-OH", pk, pv, 2);              /* 0.14 */
  const char* e[] = {"CT", "OH"};
  molrs_ff_def_type(ff, "bond", "harmonic", "CT-OH", e, 2, pk, pv, 2);        /* 0.15 */
  ```
  - The name is never split.
  - A missing style is `INVALID_ARGUMENT`; it used to be created.
  - A conflicting re-definition is `INVALID_ARGUMENT`; an identical one does
    nothing.
- **The force-field JSON format changed.**
  - `molrs_ff_from_json` requires `str_params` on every style and `endpoints`
    / `str_params` on every type. Unknown or missing keys are refused, so
    **JSON written by 0.14 is refused**.
  - `molrs_ff_to_json` writes the new shape, with `units` / `special_bonds`
    only when declared.
- **`molrs_ff_new` declares no units and no special_bonds.**
- **Removed:** `molrs_c_api_version()`, `MOLRS_C_API_VERSION` and
  `molrs_frame_schema_version()`, with no replacement.
- **f32 metadata removed.** `MolrsMetaType` loses `F32` (5), `F32x3` (13),
  `F32x6` (15) and `F32x9` (17); the remaining values are unchanged.
  `MolrsMetaValue` loses `f32_value` and `f32x9`, which changes its layout, so
  recompile.
- **`molrs_frame_meta_key(index)` uses insertion order**, not sorted order.
- **`molrs_frame_from_smiles` can return `MOLRS_STATUS_INTERNAL_ERROR`.**
- **`molrs_schema_vocab_version()` returns 2.**

### C++ (`molrs-cxxapi`)

- **Removed:** `cxx_api_version()`, `frame_schema_version()` and the
  `CXX_API_VERSION` file. Gate on `cxx_api_capabilities()` instead; its bits
  are unchanged.
- **f32 metadata removed.** `MetaType` loses `F32`, `F32x3`, `F32x6` and
  `F32x9`. Later values shift down (for example `F64` 6 → 5, `String` 7 → 6,
  `F64x9` 18 → 14), so stored integer values do not carry over. `MetaEntry`
  loses `f32_value` and `f32_values`.
- **`frame_meta_entries` returns insertion order**, not sorted order.

### `molrs-ffi`

- **No public items were removed.** The capsule names move to the `0.15` line;
  see [All surfaces](#all-surfaces).

### Also new in 0.15

- **CGsmiles:** `io::smiles::parse_cgsmiles` → `CGSmilesIR`, the fragment SMILES
  dialect, and Python `molrs.io.CGSmilesIR` with its record classes.
  `SmilesError` (Python `ValueError`) carries `kind` / `span` / `input` /
  `notation`.
- **Assembly:** the site-graph `builder::Assembler`, `SitePlacer`,
  `GrowthPlacer`, `AxisOrienter` and `perceive::Coarsener`; ports on every
  graph type; `SubgraphMatcher`; `Frame::subset`; and the `molrs.op` numeric
  base. The `Fragment` / `TracePlacer` types seen on the 0.15 development
  branch never shipped.
- **Force field:**
  - `ForceField::merge`, `empty_like`, `units` / `set_units`;
    `Typing<T>` / `Match`; `ElementTypifier`.
  - `core::store::type_labels::{TypeName, TypeLabels}`.
  - `lammps_coeff_params` / `lammps_coeff_values`; the frcmod writer.
- **Store:** nullable columns (`insert_nullable`, `validity`; persisted in
  zarr); Python `MetaDocument`.
- **Declared precision:** `store::precision::{quantum, quantize,
  quantize_in_place, check_precision, PRECISION_MIN, PRECISION_MAX}`;
  `Block::{set_precision, precision, clear_precision, precisions}`;
  `SequenceSchema::{declare_precision, precision}`. Python
  `Block.set_precision` / `Block.precision` (pickled with the block) and
  `SequenceSchema.declare_precision` / `precision`; WASM `Block.precision`;
  C++ `frame_set_precision`.
