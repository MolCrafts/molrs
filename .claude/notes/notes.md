# molrs — Evolving Decisions

Working notes captured by `/mol:note`: a decision, why it was taken, and what
it costs someone who does not know it. An entry leaves this file when it is
promoted into `CLAUDE.md`, lands in `.claude/notes/release.md` as shipped
behaviour, or is superseded — the record of what shipped is the git history,
not a pile of notes.

```
## YYYY-MM-DD — <topic>
**Decision:** <one-liner>
**Why:** <constraint, incident, or measurement>
**Status:** provisional | hardening | promoted (→ CLAUDE.md §section)
```

---

## 2026-09-14 — regions are solids with a signed distance; one type per shape

**Decision:** `spatial::region::Region` requires `bounds` + `distance`
(negative inside, positive outside, zero on the boundary) and derives
`contains_point` / `contains` / `distance_grad` from it. Every shape describes
its inside; outside, shells and voids are `NotRegion` / `AndRegion` /
`OrRegion`. `HollowSphere` is gone (`Sphere & ~Sphere`). New shapes:
`HalfSpace`, `Cylinder`, `Ellipsoid`, `Polyhedron` (a watertight `TriMesh`,
whatever produced it), `SphereUnion` (centres + radii, minimum image on the
box's periodic axes; with `r = r_vdW + r_probe` its complement is the
solvent-accessible void). molpack's own `Region` trait, `StlRegion` and BVH
move here; molpack keeps only its penalty policy.
**Why:** three parallel region models (molrs boolean, molpack SDF + mesh,
molvis TS ball field) for one concept; the PEO-in-void case needs a region
built from atoms in memory, no mesh file, and a lattice mask that asks the
region about ~10⁶ sites — a BVH per query, not a scan.
**Status:** provisional

## 2026-09-07 — cargo needs `target/` on local disk; a symlink, not `CARGO_TARGET_DIR`

**Decision:** make `<repo>/target` a symlink to node-local storage. Do **not**
solve this with `CARGO_TARGET_DIR` — several gates hard-code `target/...`
paths and will read a different tree than the one they just wrote.
**Why:** the checkout is on Lustre (`/nobackup`, and `/home` too), where cargo
blocks forever on its own artifact lock. Measured twice: `cargo doc` at 9 h 27 m
elapsed with 0 s of CPU, and a pre-push `cargo test --lib` at 33 min with 1 s of
CPU — no `rustc` child in either case, `wchan = ldlm_flock_completion_ast`, and
`lsof` showing the blocked process as the *only* holder of
`target/debug/.cargo-artifact-lock`. So it is a wedged flock, not a slow
fingerprint scan over a large artifact tree.

`CARGO_TARGET_DIR=/tmp/...` does unblock cargo, and that is what an earlier
version of this note recommended — but every committed path (`target/wheels`,
`target/release/libmolrs_capi.so`, the CMake `CARGO_TARGET_DIR` default) assumes
the one shared target dir; the symlink keeps them all pointing at one real
directory.

```bash
mv target target.lustre-cache        # keep or delete; it is only a cache
ln -s /tmp/molrs-target target
```

Node-local means node-local: on a different login node the symlink dangles and
the cache is cold. That is the cost, and it is smaller than losing a day to a
lock that never returns.
**Status:** hardening

## 2026-09-07 — `dump local` accepts every `dump_modify label`

**Decision:** the LAMMPS `dump local` reader accepts `ENTRIES` (the default)
plus `BONDS` / `ANGLES` / `DIHEDRALS` / `IMPROPERS` / `NEIGHBORS`, and records
which one in `frame.meta` as `dump_local_label`. Rows still land in `entries`
whatever the label says.
**Why:** the reader's own doc comment cited OVITO's LAMMPS-dump-local manual
while implementing one of the six labels it lists — and `dump_modify …
label BONDS` is precisely what that manual tells users to set, so the
recommended setup was a hard parse error. The label is also the *only*
meaning-bearing part of such a file: column names are whatever the dump
command was given, and the default is `c_bond[1] c_bond[2]`, which says
nothing. Consumers must read the label rather than re-guess from the header.
The rows keep the `entries` name because the label says what they mean, not
that they satisfy a contract-bearing block's schema — the argument recorded
at the `block_name` binding still holds.
**Status:** provisional

## 2026-09-07 — `Frame.getMeta` completes the wasm meta pair

**Decision:** `molrs-wasm` `Frame` exposes `getMeta(name) -> string | undefined`
alongside the existing `setMeta` / `getMetaScalar` / `setMetaScalar`. It
returns `undefined` for non-string values rather than stringifying them.
**Why:** `setMeta` had no reader, so a word-valued label written through it
was unreachable from JS — `getMetaScalar` returns `undefined` for anything
that does not parse as a number. Found while wiring `dump_local_label`
through to molvis. Not stringifying numeric meta keeps `"1"` and `1`
distinguishable at the boundary.
**Status:** provisional

## 2026-09-04 — amber-prmtop-complete-02 debts found at spec time

**Decision:** record, do not fix in this phase. (A third item — docs claiming
no `molrs/tests/` tree while `architecture_gate.rs` sits in it — was fixed on
2026-09-17.)
**Why:** iron-law naming of rot that is out of this spec's layer or below the extract-on-second-use bar.
**Status:** provisional

1. ~~σ/ε closed form duplicated between `ff/forcefield/readers/prmtop.rs` and `io/data/prmtop_tables.rs`.~~ Closed 2026-09-17: both call `core::math::pair_form::lj_ab_to_sigma_epsilon`. Two call sites *is* the extract bar — CLAUDE.md § Prefer says "extract only at a second call site", which this note had mis-stated as the third.
2. `forcefield/gaff.rs` cited `scripts/gen_gaff_energy_oracle.py` for the `AMBER_COULOMB` sander measurement; that generator is not in the tree. The value stands (`18.2223²`); the rustdoc no longer points at the missing script.


## 2026-08-26 — development tools track latest; other-platform UB out of scope
**Decision:** rustc/clippy/rustfmt = `rust-toolchain.toml` `channel = "stable"`
and CI `dtolnay/rust-toolchain@stable`. wasm-opt = latest binaryen GitHub
release. uv action = latest major. pre-commit-hooks = latest tag. Never pin a
compiler/linter minor to hide CI drift. Other-platform UB (Windows/macOS-only)
is out of scope until 0.14 lands on `dev`.
**Why:** pinning rustc 1.96 made local prek green while GHA clippy on 1.98 was
red; the next rustc bump would just repeat it.
**Status:** active

## [2026-08-10] Binding-surface symmetry (settled after the neighborlist chain)

The quality of the facade (the public API) outranks the internal implementation;
internals are refactored incrementally, without chasing a single sweep and without
blocking a release.

**Rule**: the Rust / Python / WASM surfaces must stay symmetric — same names
(`NeighborList` the engine / `Neighbors` the table), same shape (build / update /
neighbors + the Option column semantics), same defaults (`FULL`). Before adding to
or changing any one binding surface, check it against the other two.

Known asymmetries (in internal-refactor priority order):

1. **The wasm `NeighborQuery` symmetry gate is deferred to 0.15** (2026-08-25).
   Deletion is not among the options: in-tree consumers are
   `compute/hbond/detect.rs` (`from_columns` / `free_columns`,
   `QueryMode::CrossQuery`), `compute/rdf/mod.rs`, `compute/dynamics/van_hove.rs`
   and `ff/potential/soft.rs`. wasm has no consumer yet (facade-first), so 0.14
   neither adds the symmetry gate nor deletes the engine type.
2. The `LinkedCell` / `BruteForce` aliases survive only for the molvis link
   (default `FULL`, safe); once molvis moves to the engine API they are **deleted**
   — two doors are not maintained long-term.
3. The remaining routed items are done slowly, as needed: core SoA
   `update_columns`, splitting `neighbors/mod.rs` into `table.rs` (a pure move).
   Borrowing `Compute::Args` is done (2026-08-10).

**Status:** active

<!-- mol:note:topic:md-experimental-ship-0.14 -->
## 2026-09-20 — no BLAS feature; the 3x3 helpers are closed-form only

**Decision:** the `blas` feature and `ndarray-linalg` are gone. `core::math`
keeps the hand-written `det3` / `inv3` and a single-body `matmul` that `rayon`
parallelises by row.
**Why:** the LAPACK path served only a 3×3 determinant and inverse — slower
than the cofactor forms next to it — and it `.expect()`-panicked on a singular
matrix where the default build returned a value, so enabling an
"optimisation" feature changed panic semantics. No consumer in any sibling
repo enabled it. The 2026-05-28 entry about backend selection is superseded.
**Status:** locked

<!-- mol:note:topic:cgsmiles-reader-shape -->
## 2026-09-21 — `parse_cgsmiles` is one public door over private steps (Shape check #2 bent, recorded)

A `CGSmilesIR` that exists is fully instantiated and validated (and, once
`cgsmiles-01d` lands, fully resolved): port pairing is resolved once, by the
reader, and pairing at an intermediate level is only definable over an
instantiated level, so an un-instantiated IR is not a state any caller may
hold. Exposing `parse` / `instantiate` / `validate` as separate public
primitives would make that illegal state representable.

**Rule**: keep `io::smiles::parse_cgsmiles` the single public entry for the
notation; `split_blocks`, `parse_block`, `parse_body`, `check_coverage`,
`instantiate`, `validate_ir` (and 01d's `resolve`) stay private free functions
in `io/smiles/cgsmiles/`, never methods on `CGGraph` and never re-exported.
`fragments[k]` is the authority for a fragment's shape; `levels[k+1]` is its
expansion, produced once by the reader.

<!-- mol:note:topic:cgsmiles-descriptor-order -->
## 2026-09-21 — a descriptor's bond order is the symbol adjacent to its bracket

Matches the CGsmiles reference (`read_fragments.py:148-157` @ 910c9ee), which
processes brackets one at a time. The mid-chain form `C[$]=CC` (order `None`
on the descriptor, C0=C1 double) is reference-verified, not an inference.

**Rule**: a bond symbol preceding a descriptor run belongs to the run's first
descriptor; a symbol following a leading run belongs to its last; mid-chain a
symbol after a bracket is an ordinary bond to the next atom. `[<=1]` (BigSMILES
v1.0 in-bracket order) is `BondInsideDescriptor`, never accepted silently.

<!-- mol:note:topic:cgsmiles-writer-single-order -->
## 2026-09-21 — `write_fragment_smiles` writes an explicit single order as `-`

`Some(BondKind::Single)` and `None` are different IR values: 01d's pairing rule
promotes a bond between two written-aromatic ports to aromatic only when no
symbol was written (Daylight, biphenyl's `-`).

**Rule**: the writer's omit-default-single policy applies to chain bonds only;
a descriptor order is always emitted when `Some`, so `CC-[$]` round-trips.

<!-- mol:note:topic:cgsmiles-ring-markers -->
## 2026-09-21 — CGsmiles ring markers: any digit run after `%`, marker 0 valid

Grünewald et al., JCIM 2025 (DOI 10.1021/acs.jcim.5c00064) §2.1.4: unlike
OpenSMILES, `%123` is one marker. OpenSMILES §3.4: marker 0 is valid and a
bond symbol may sit at either end of a closure.

**Rule**: `%` + any digit run is one marker (`u16`; overflow →
`CgInvalidRingMarker`, as is a bare `%`); `0` and `%00` are valid; a symbol at
one end sets the ring order, the same symbol at both ends is fine, differing
symbols are `RingBondConflict`. The SMILES parser's own `%n` handling still
reports `UnexpectedEnd` (see cgsmiles-deferred-fix).

<!-- mol:note:topic:cgsmiles-port-vs-descriptor -->
## 2026-09-21 — `io::smiles::DescriptorKind` and `core::PortKind` are two enums by design

Same four roles (`$ < > !` → Symmetric / Left / Right / Shared), two homes: the
AST names what was written, `core` names what is stored — the same split as
`BondKind` vs `BondType`/`BondNumber`.

**Rule**: the only conversion site is `io/smiles/cgsmiles/to_fragment.rs`
(cgsmiles-02b); `core` never names `io`, and no second `DescriptorKind → PortKind`
mapping may appear. `PortKind`'s stored form is the glyph (`Str` column
`port_kind`), a recorded departure from `BondType::code()`.

Resolved at the Python seam (user decision 2026-09-21): **the glyph is the one
spelling for the four roles everywhere a user reads or writes them** —
`BondingDescriptor.kind` in `molrs.io` crosses as `"$" | "<" | ">" | "!"`,
the same string a stored port's `port_kind` column and `views.Port` carry;
`DescriptorKind::as_str()` and `PortKind::as_str()` return the same glyphs. The
01e rule "a name enum crosses as its lowercase name" still holds for
`BondingDescriptor.order`, `ResolvedPair.kind` and `PairEnd.end`, which have
no notation glyph. The two Rust enums stay separate (AST vs core).

<!-- mol:note:topic:cgsmiles-frag-id -->
## 2026-09-21 — `frag_id` is the provisional per-atom fragment-instance key

Mirrors the reference implementation's `fragid`. `mol_id` groups atoms into
molecules (a fragment instance is sub-molecular); `res_id` is the biopolymer
residue key earmarked for the pending schema-vocabulary spec.

Landed 2026-09-21 (cgsmiles-02a): `core::Fragment` is the third `MolGraph` leaf;
`Atomistic`/`CoarseGrain`/`Fragment::try_from_molgraph` resolve their standard
kinds through `MolGraph::try_register_kind(name, arity)` — by name with an arity
check that returns `MolRsError::validation` instead of `register_kind`'s assert,
replacing the `unwrap_or(KindId(0..3))` aliasing that let a foreign
first-registered kind pose as `bonds`. `mapping.rs` (`CGMapping`,
`WeightScheme`) is deleted.

**Rule**: write fragment-instance membership as the open node prop `frag_id`
(`Int`), all-or-nothing per Frame column; do not reuse `res_id` or `mol_id`.
Two writers exist by construction: `CGSmilesIR::to_atomistic` stamps it on
every expanded atom (`cgsmiles/to_atomistic.rs`, landed in cgsmiles-01d) and
02a's `Fragment::set_frag_id`; the schema-vocabulary spec gives the key one
validated owner.

<!-- mol:note:topic:cgsmiles-v1-refusals -->
## 2026-09-21 — CGsmiles v1 refusals and conventions

**Rule**: refuse, never drop, these notation features in v1: `[!]` squash
(`CgSquashUnsupported`), non-default `w` weights and chirality `x`
(`CgUnsupportedAnnotation`), order-0 `.` bonds (`CgInvalidBondOrder`),
atom-level `;` annotations (`AtomAnnotationUnsupported`, raised by the fragment
SMILES dialect, not by a second lexer). `CGNode.charge` is a partial charge in
`e` by molrs convention (the notation states no unit). A wildcard bead `[#*]`
resolves like any other name and is `CgUndefinedFragment("*")` once a fragment
table follows. The last block of a multi-block string is atomistic by position
(no flag).

<!-- mol:note:topic:cgsmiles-deferred-refactor -->
## 2026-09-21 — deferred to `/mol:refactor` (found by the cgsmiles chain)

- `io/smiles/parser.rs` and `perceive/smarts/` each parse SMARTS with their own
  AST; `io::smiles::parse_smarts` has zero non-test in-repo consumers, but the
  cross-repo audit (molpy, molpack, Atomiverse, binders) has not run — do not
  delete before it does.
- `io::smiles` names a notation family yet contains an inner `smiles` module
  (`#[allow(clippy::module_inception)]`).
- `pub mod error` / `chem` / `smiles` under `io/smiles` give every entry point
  two public paths; the flat re-export is meant to be the only one.
- Pre-existing over-limit functions in touched files: `parse_atom_primitive`
  (237 lines), `write_primitive`, `build_tree`, `build_recursive_env`;
  `SmilesErrorKind::message` grows with every notation.
- `cgsmiles/resolve.rs::last_level_ports` and `cgsmiles/to_atomistic.rs::to_atomistic`
  repeat the same definition lookup (`defs.get` → refuse a coarse body →
  `FragmentCache::get_or_build`) with different error kinds; second call site,
  extract when a third appears or when the error-kind split is decided.
- `FragmentCache` lives in `resolve.rs` but is consumed by `to_atomistic.rs`;
  promote to `cgsmiles/fragment_cache.rs` on a third caller.
- Two `Port` structs: the public `core::system::fragment::Port` (02a) and 01d's
  private port-table entry `cgsmiles/resolve.rs::Port`; different modules, no
  conflict, but one name for two things — rename the private one.
- `BondType` has no quadruple variant (`core/system/bond.rs:28-38`), so
  `BondKind::bond_type` maps `Quadruple` to `Double` (the number stays
  `BondNumber::Quadruple`); widening `BondType` is its own spec.
- Python enum conventions are split: `io` crosses names as lowercase strings
  (`build_smiles_emit_options`, `cgsmiles.rs`), `core` crosses bond classes as
  two `u32` codes (`set_bond_class` / `bond_type`, `molgraph.rs:808,829`) that
  have no code for `up`/`down`/`any`/`ring`. Reconciling is a breaking Python
  API change with its own spec.
- One count, four spellings at the Python seam: `n_nodes` / `n_atoms` / `n_beads` /
  `n_ports`; and `views.py` now carries four `__init__`/`__reduce__` monkey-patch
  pairs (one per leaf) that a single shadow-class lookup could replace.
- `_lib.pyi` parity is guarded at class-name level only
  (`tests/test_stub_parity.py`); method- and parameter-level parity, which the
  stub header claims, is unguarded. No doctest runner executes the binder's
  `>>>` examples (`pyproject.toml` `testpaths` only), and no `ruff` step is wired
  into any gate, so Python-side drift is invisible.

<!-- mol:note:topic:cgsmiles-deferred-fix -->
## 2026-09-22 — the cgsmiles deferred-fix list is closed

Every item the chain deferred is fixed, in four same-day commits after the
0.15.0 bump. Kept as the record of what the list contained and what it cost.

Closed on 2026-09-21: `[#6]` via `Element::by_number`; SMILES `%n` →
`InvalidRingMarker`; the six discarded `Result`s and the `InvalidElement`
mislabel on the SMILES build path (→ `SmilesErrorKind::Build`); unmatched ring
closures and unknown bracket elements refused at parse; the CGsmiles
repeat-count cap; the coarse-body scanner bounded by the body;
`remove_hydrogens` bonds-only degree with a port-handle exemption; the `1.008`
mass literal; `add_hydrogens` / `remove_hydrogens` / `Perceive::find_hydrogens`
returning `Result`; `read_frame` refusing an unregistered relation block, a
registered block missing an endpoint column, a dtype-conflicting relation prop
and an endpoint index past the atoms block; `MolGraph::merge` returning
`Result`; nullable Frame columns end to end (`Block::insert_nullable` /
`validity`, `to_frame` emitting masks, `read_frame` honouring them, the
all-or-nothing `frag_id` rule gone, Python `Block.validity` /
`insert_nullable` with masks through `copy` and pickling);
`molrs.io.SmilesError`; the binder's private intra-doc links;
`views.Atomistic.def_bond`.

Closed on 2026-09-22:
- **Zarr persists validity masks.** A mask is a `bool` array at
  `<block>/_validity/<column>`, a reserved **subgroup** of the block group,
  one flag per row, for a frame group and for a trajectory alike. The reader
  skips non-array children of a block, so a pre-mask reader — molrec, molvis —
  ignores the subgroup instead of reading it as a column, and an old store
  still reads as fully valid. The name is `_validity`, not `__validity__`:
  Zarr V3 reserves the `__` prefix and `zarrs` enforces it. `ColumnSchema`
  gains `nullable` (`#[serde(default)]`, absent when false), `from_frames`
  unions it, `SequenceSchema::declare_nullable` is the public opt-in for a
  hand-declared pin, and appending a masked column to a pin that declares it
  non-nullable is refused. `same_block` compares masks, so a frame repeating
  its values under a moved mask still earns its update.
- **No write path asserts an invariant by a panic.** `MolGraph::add_node_with`
  and `to_frame` (with the `Atomistic` / `CoarseGrain` / `Fragment` delegates)
  return `Result`; `to_frame`'s was a live process kill, not style debt.
- **The class is closed at the door.** `coerce_canonical` no longer lets a
  value through whose element type cannot be stored at the key's declared
  dtype: a `Str`/`Bool` under a numeric key, a number under a string key, an
  `F64` under a `UInt` or `Int` key are refused by `set_node` / `set_atom`,
  which already returned `Result`. That is what makes the seven `.expect`s in
  the infallible leaf constructors (`Atomistic::add_atom{,_xyz,_bare}`,
  `Fragment::add_atom_{xyz,bare}`, `CoarseGrain::add_bead{,_bare}`)
  unreachable rather than merely fallible, and it is why `to_frame`'s schema
  arm now has only one caller left that can trigger it — `node_table_mut()`,
  the raw door past the property API.
- **Long functions split**, pure moves: `Block::merge` 106 → 28,
  `Block::sort_indices` 82 → 12, `parse_atom_primitive` 224 → 18,
  `generate_3d_impl` 206 → 56 (an A/B harness over 100 SMARTS parses and 26
  ETKDG runs produced byte-identical output, coordinates at full precision).
- The `|n` repeat-count refusal names its bound; `molrs.frame.Block` no longer
  defines seven methods twice.

**Rule**: a value whose element type contradicts a key's declared dtype is
refused where it is written, never carried to the Frame boundary. A write path
returns its invariant; it does not assert it with `expect` or `unreachable!`.

Still open, and none of it is cgsmiles debt:
- `io/zarr/sequence.rs`: `FrameSequenceWriter::commit` (~235 lines) and
  `FrameSequence::assemble` (~100) are over the 80-line rule — pre-existing,
  `/mol:refactor`, and the largest such functions left in the crate.
- `molrs-python/python/molrs/frame.py` is not ruff-governed (~470 pre-existing
  findings; never run `ruff format` on it — it rewrites unrelated lines).
- `MolGraph::node_table_mut()` writes a column without consulting the key's
  declared dtype, so it can still build what the property API now refuses.

<!-- mol:note:topic:cgsmiles-pairing-order -->
## 2026-09-21 — descriptor pairing: parse-order edges, per-atom entities, greedy scan

The CGsmiles reference (`resolve.py` @ 910c9ee `match_bonding_descriptors`)
scans source entities × target entities × source descriptors × target
descriptors, first free compatible pair wins, both consumed. Its entities are
graph nodes: child beads at an intermediate level, **atoms** at the atomistic
level. Grouping an instance's whole port list as one entity changes the scan
order and can bond a different atom (`{[#A][#B]}.{#A=[$][>]CCC,#B=[<]CCO[$]}`
pairs `>`/`<` on C0–C0 in the reference, `$`/`$` on C0–O under per-instance
grouping).

**Rule**: `cgsmiles/resolve.rs` builds one port entity per child node at an
intermediate level and one per port-carrying atom (walker order) at the last
level; edges are iterated in **parse order** (ring closures last), which is
where molrs and the reference (networkx adjacency order) legitimately differ —
only the unlabelled connectivity of the expansion is promised isomorphic, not
atom indices. Compatibility is `flip(kind) == kind && label == label &&
effective order == effective order` with `flip` a total involution
(Left↔Right, Symmetric, Shared); `None` ≡ `Single`, so `[$]` pairs `-[$]` and
`=[$]` never pairs `[$]`. An edge with no free compatible pair is
`CgUnmatchableEdge { level, edge }`, never a silent skip.

<!-- mol:note:topic:cgsmiles-bond-class-precedence -->
## 2026-09-21 — inter-fragment bond class follows Daylight, not the reference's 1.5

The reference sets bond order 1.5 whenever both endpoint atoms are aromatic,
even over a written symbol (biphenyl `-[$]` and `=`/`=` both become 1.5
there). molrs has no 1.5: `BondKind::Aromatic` maps to `BondType::Aromatic` +
`BondNumber::Unknown`, because the notation declares delocalisation, not a
Kekulé phase.

**Rule**: a written descriptor order wins; absence between two *written*-aromatic
ports (the `is_aromatic` stamp `Builder` wrote, read off a freshly converted,
never-perceived body — the `FragmentCache` no-perception invariant) is
`Aromatic`; everything else is `Single`. A Kekulé-spelled ring
(`[$]C1=CC=CC=C1`) therefore bonds `Single`. A coarse edge of multiplicity n
is n separate bonds, never a multiple bond (Martini cyclohexane
`{[#SC3]=[#SC3]}` → two singles).

<!-- mol:note:topic:cgsmiles-python-seam -->
## 2026-09-21 — `molrs.io.CGSmilesIR` is the one Python door; nested IR records are frozen values

In this binding "Reader" means a lazy, path-backed trajectory cursor
(`XYZTrajReader`, `DCDTrajReader`, …), so a text-in/IR-out parser is not a
Reader; the reader-shaped `CGSmilesReader(text).read()` is molpy's to write
over this class, as molpy's `SmilesReader` wraps `molrs.io.SmilesIR`.

**Rule**: `molrs.io.CGSmilesIR(text)` mirrors `PySmilesIR` (`{inner, input}`,
`#[new]`, `to_atomistic()`, `__repr__`); no free `parse_cgsmiles`, no
`n_levels` (`len(ir.levels)` is the fact). The seven nested records follow
the `LammpsLog` house style: `frozen, skip_from_py_object`, getters only, no
`#[new]`, values handed out by cloning. An enum that *is* a count crosses as
the count (`CGEdge.multiplicity`, no `order`); a name enum crosses as its
lowercase variant name through a total mapping fn (no `_ =>`) — except the
descriptor kind, which crosses as its notation glyph like `port_kind`; a sum whose
payload types already distinguish the variants has no tag (`CGFragmentDef.body`
is a `CGGraph` or a `SmilesIR`; `CGEdge.derived_from` is `(level, pair) | None`),
one whose payloads do not gets a tagged pyclass (`PairEnd.end/index/port`).
Errors go through the one `smiles_error_to_pyerr`. `tests/test_stub_parity.py`
keeps `_lib.pyi` and `molrs._lib` equal at class-name level with
`inspect.ismodule` as the only exemption — never an allowlist.

<!-- mol:note:topic:trait-principle-1-static-dispatch -->
## 2026-09-21 — trait principle 1 (provisional amendment): static-dispatch bounds may name `Self` or be generic

`architecture-rules.md` § Trait design states principle 1 absolutely ("no `Self`
in return position, no generic methods"). Three in-tree traits contradict it and
are never used as trait objects: `io::reader::FromFrame` (`Self`-returning
constructor, `io/reader.rs:69`), `FrameAccess::visit_block<R>` (generic method,
`core/store/frame_access.rs:46`), and `conformer::ElementGraph` (cgsmiles-02c:
`try_from_molgraph(MolGraph) -> Result<Self>`, the bound of
`Conformer::generate<M: ElementGraph>`).

**Rule** (provisional until promoted into `architecture-rules.md`): a trait must
be object-safe when it is used as a trait object; a trait that exists only as a
static-dispatch bound beside its one consumer may name `Self` in return position
or carry generic methods. `ElementGraph` lives in `conformer/element_graph.rs`
because that generic function is its only consumer; `CoarseGrain` deliberately
does not implement it (bead types are not elements).
