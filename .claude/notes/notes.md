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
3. **`PotentialCompiler.defer()` is Python-only, and wasm has no `PotentialCompiler`
   class** (system-forcefield-04, 2026-09-25). Rust callers compile when they hold the
   frame; wasm compiles inside `typifier.toPotentials(frame)` and `LBFGS`. Add the door
   when a consumer needs it.
4. **The backmap primitives are bound for Rust and Python only**
   (backmap-primitives-07, 2026-09-26): `SubgraphMatcher`, `center` on
   `Atomistic` / `CoarseGrain` / `Fragment`, `Fragment.link`, `Fragment.merge`,
   `Frame.subset`, `CGSmilesIR.to_coarsegrain`; also `molrs.op` and
   `UnitRegistry.define_lj_sigma`. Python-only door: `Fragment.to_atomistic()`
   (Rust: `Atomistic::try_from_molgraph(fragment.into_inner())`). Python-only
   column access: `frame["atoms", "mol_id"]` (tuple = (block, key) on `Frame`;
   on `Block` a tuple means several columns). Name parity: `center(group)` in
   both. Recorded shape asymmetry: `Frame.subset(mask, block="atoms")` in Python
   vs `Frame::subset(block, rows)` in Rust. wasm and C/C++ add them when a
   consumer needs them.
5. The remaining routed items are done slowly, as needed: core SoA
   `update_columns`, splitting `neighbors/mod.rs` into `table.rs` (a pure move).
   Borrowing `Compute::Args` is done (2026-08-10).

**Status:** active

<!-- mol:note:topic:md-experimental-ship-0.14 -->
## 2026-09-20 — no BLAS feature; the 3x3 helpers are closed-form only

**Decision:** the `blas` feature and `ndarray-linalg` are gone. The 3×3
`det3` / `inv3` are hand-written closed forms, now in `op::linalg` (2026-09-26,
assembly-01). The `matmul` clause is superseded: `matmul` had test-only use and
is deleted.
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
Two writers exist by construction: `MolGraph::replicate` stamps it column-wise
on every copy
(assembly-03; `CGSmilesIR::to_atomistic` reaches it through `replicate`),
and `Fragment::set_frag_id` per atom. The key is
spelled by one crate-level constant until the schema-vocabulary spec gives it
one validated owner.

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

---

## 2026-09-22 — the build gate is a filter problem, not a crate-size problem

**Decision:** every root-workspace cargo call goes through a `cargo mrs-*`
alias (`.cargo/config.toml`); scoped test runs are `cargo mrs-test -- <module>`,
which narrows the **filter** and never the feature list (`scripts/test-scope.sh`
was deleted 2026-09-25); `[profile.dev] debug = "line-tables-only"` in all six roots; hooks are
scoped with `files:` instead of `always_run: true`; `target/` gets pruned when
it passes ~20 GB. Full note: `.claude/notes/build.md`.

**Why:** measured on the 4-core Lustre box, `molcrafts-molrs` at 293k lines /
2513 tests. Running the suite was never the cost (5.3 s). The costs were
(a) `test_single: "cargo test {path}"`, which resolved a *different* feature
set and so rebuilt the whole crate on every single-test run, and (b) a
`target/` that had grown to 101 GB — the same one-file edit cost 44-79 s there
and 6-7 s in a clean one, because a 1.4 GB incremental cache has to come back
over Lustre whenever it falls out of page cache. `debug = "line-tables-only"`
takes that cache to 420 MB.

**Rejected, with the measurement that rejected it:** splitting `ff/params`
(107,883 lines, 37% of the crate) into its own crate. Compiled standalone,
those 19 files build in **1.5 s** — static tables are ~2% of the build time
despite being 37% of the lines. Line count is not where rustc spends time
here, so neither that extraction nor the wider module-crate split (worst case
today: 27 s when `lib.rs` itself changes) pays for the churn. The
"Single crate (0.12+)" rule in `architecture-rules.md` stands.

**Status:** promoted (→ CLAUDE.md § Build cache, § Build & Test Commands)

---

## 2026-09-22 — frame-meta binder order, found debt

**Decision:** Python, C, C++ and wasm enumerate `frame.meta` in insertion order. Three leftover costs are named and not fixed in that change.

**Why:**
- The `molrs-capi` Rust `#[cfg(test)]` suite never runs. Evidence: `molrs-capi/src/schema.rs:150-168`. Pre-push and `ci-capi.yml` build and `ctest`; they do not `cargo test --manifest-path molrs-capi/Cargo.toml`. Route: `/mol:fix`.
- `molrs_frame_read_meta`, `molrs_frame_meta_count` and `molrs_frame_meta_key` each `clone_frame` the whole frame. `Store::with_frame` (`molrs-ffi/src/store.rs:117`) is the borrow-only door. Route: `/mol:refactor`.
- `PyFrameMeta::map` cloned the whole `MetaMap` on every read. Route: `frame-meta-dict-parity-04-dict-views`, which deletes it.

**Status:** provisional

---

## 2026-09-22 — per-frame Zarr groups and MolRec carry meta untagged

**Decision:** a per-frame Zarr group (`molrs/src/io/zarr/frame_io.rs:536` attribute write, `:640` attribute read) and `MolRec::meta` (`molrs/src/core/store/record.rs:100`) carry meta untagged. A dtype does not survive either path. Making those paths tag-preserving is a separate 0.15 change, not this binder rule and not a compatibility deferral — and not a later minor.

**Why:** `write_frame_group` stores `to_attr_value()`, the plain payload, in the group's attributes, and `read_frame_group` rebuilds each entry with `MetaValue::from_attr_value`, which infers. `MolRec::meta` is a `JsonMap<String, JsonValue>`. A tag is durable only on a declared sequence schema and on the serde frame document; these two paths were already outside that bound.

**Status:** provisional

---

## 2026-09-22 — frame.meta freezes documents; found debt

**Decision:** Every `frame.meta` door returns a frozen value (`tuple` for a fixed-length vector or a JSON array, `molrs.MetaDocument` for a JSON object). `MetaValue.value` stays a plain decode so pickle keeps working. Document key order stays unspecified. Two molrec call sites are molrec's to fix on this same 0.15 tree — named here, not edited, and not gated on a tag.

**Why:**
- `frame.meta["run"]["step"] = 3` used to mutate a decoded snapshot and vanish. A `MetaDocument` raises `TypeError` instead. `copy()` is the unfreeze, and `json.dumps(frame.meta["run"].copy())` is the JSON idiom.
- `molrec/src/molrec/core/bindings/zarr.py:783-785` does `json.dumps(frame.meta[key])` when `series.dtype == "json"`. A document is not a `dict`, so this raises. molrec's fix: `json.dumps(value.copy())`, or `frame.meta.typed()[key].value` (plain by the value-object rule). Same 0.15, not a post-publish deferral.
- `molrec/tests/molrs_adapter.py:259-261` reads `value.dtype` off `dict(frame.meta).items()`. `dict(meta)` has not returned tagged values since that door started handing out plain values — the adapter is already stale, independent of the freeze. molrec's to fix. `molrec/tests/molrs_adapter.py:110-113` round-trips `dict(frame.meta)` and does not need a tag.
- Bulk doors that assign the mapping back still round-trip, verified by reading them and not edited: `Frame.to_dict` (`molrs-python/python/molrs/frame.py`, `dict(self.meta)`, read-only), `Frame.copy` (`new.meta = self.meta`, whole-map fast path), `molpy/src/molpy/io/data/pdb.py:66` (`out.meta = dict(frame.meta)`), `molpy/src/molpy/io/forcefield/amber.py:74` (`{**frame.meta, **dict(structure.meta)}`). Tuples and documents re-infer; the copy path never decodes.
- `serde_json` document order depends on a transitive crate enabling `preserve_order`. Neither `molrs/Cargo.toml` nor `molrs-python/Cargo.toml` declares it. Declaring it would swap every `Map` in molrs from `BTreeMap` to `IndexMap`, a core behaviour change. Order inside a `MetaDocument` is unspecified on purpose. Route: `/mol:note` (this entry). Not a compatibility shim.
- `PyFrameMeta` declares `module = "molrs._lib"` while `PyMetaValue` and `MetaDocument` declare `module = "molrs"`, though all three are exported at `molrs.*`. Observed, unowned, not fixed — changing `FrameMeta.__module__` is a visible `repr` change with no consumer need.
- `_lib.pyi` parity (`molrs-python/tests/test_stub_parity.py`) is class-name level only, so the corrected frozen return types are unguarded. Pre-existing; already on the deferred list.
- Three copies of `json_to_py` / `py_to_json`: `molrs-python/src/core/store/frame.rs`, `molrs-python/src/core/store/record.rs:297`, `molrs-python/src/io/mrec.rs:702`. Record and mrec payloads are owned values a caller re-submits wholesale, not live views, so they stay plain. The triplication is rot. Route: `/mol:refactor`.

**Status:** provisional

## 2026-09-25 — C force-field params are numeric; strings travel only through JSON

`molrs_ff_def_style` / `molrs_ff_def_type` / `molrs_ff_def_type_at` take `const char**`
keys and `double*` values, so a C caller cannot define a string param (`mixing`,
OPLS provenance) through the `def_*` calls. `ff_to_json_string` / `ff_from_json_string`
carry string params (`str_params`), endpoints and `special_bonds`. Known C-vs-Rust/Python
asymmetry (system-forcefield-01); a string-valued C door is added only when a C consumer
needs one.

## 2026-09-25 — owed: the compiled MMFF tables have no source-equivalence test

`ff/forcefield/xml.rs` claimed the shipped MMFF set was "checked field for field, at
zero tolerance, against the pre-conversion parse in `tests/ff/tables_equivalence.rs`".
That test never existed (no `tests/` tree); the only test in `ff/params/mmff.rs` is
`sorted_invariants`. Nothing checks the compiled MMFF tables against their source
values. Owed — route `/mol:fix` (a unit test in `ff/params/mmff.rs` pinning a sample of
rows per table against hand-copied MMFF94 source values). Found by system-forcefield-01.

## 2026-09-25 — input-free constructors over `ff/params` tables may `expect` a definition result (scoped amendment)

Amends 2026-09-22 "a write path returns its invariant", which is about runtime input.
The infallible public constructors over compile-time tables keep their signatures and
`expect` the result of a fallible private body that one unit test proves `Ok`; the
`expect` message names that test by path:

- `ff::typifier::opls::embedded::force_field` → `try_force_field` —
  `ff::typifier::opls::embedded::tests::force_field_defines_without_conflict`
- `ff::typifier::mmff::embedded::force_field` → `try_force_field` —
  `ff::typifier::mmff::embedded::tests::force_field_defines_without_conflict` (both tables)
- `ff::typifier::gaff::candidate_forcefield` → `try_candidate_forcefield` —
  `ff::typifier::gaff::tests::candidate_forcefield_defines_without_conflict` (GAFF, GAFF2);
  memoised per set as `GaffTypifier`'s library (moved from `ff::forcefield::gaff` by
  system-forcefield-08)
- `UFFTypifier::new` → `try_new` — `ff::typifier::uff::tests::new_defines_without_conflict`
- `ff::typifier::opls::embedded::typing_meta` → `try_typing_meta` (the join of
  `OPLSAA_TYPING` to `OPLSAA_ATOMS`) —
  `ff::typifier::opls::embedded::tests::typing_meta_joins_every_rule` (opls-gromacs-02)

Anything taking runtime input (readers, `from_xml_str`, `GaffTypifier::r#match`) returns
`Err`. Checked after system-forcefield-07: the symbols and test paths above are
unchanged; `mmff::embedded::force_field` is now called once per variant, memoised in a
`OnceLock<Arc<…>>` (`typifier/mmff/embedded.rs:40-41`). 08 moved the candidate builder
and rewrote the GAFF line above.

## 2026-09-25 — GROMACS force-field files model a molecule, not directives (resolved by opls-gromacs-01)

Both GROMACS force-field files used to write/read bonded rows by atom index into an
`[ atoms ]` table instead of the force-field directives. opls-gromacs-01 removed the
molecule model: `GromacsTopFfReader` and `GromacsTopFfWriter` deal only in `[ defaults ]`,
`[ atomtypes ]` (split into `atom/full` + the `pair/lj/cut` self row), `[ bondtypes ]`,
`[ angletypes ]` and `[ dihedraltypes ]`, keyed by type labels (`X` ↔ the empty
wildcard). The reader refuses every molecule section by name ("topology: read with
io::data::top::read_top") unless the caller skips it with `with_skipped_directive`. Its
debts are gone with it: the writer no longer invents `[ atoms ]` fields (`resnr 0`,
`residu LIG`, `cgnr 1`, `charge` / `mass` `0.0`), per-instance `TypeConflict` from
bonded rows cannot arise, and no `[ atoms ]` sections are merged across molecule types.
A full `.top` now needs its molecule sections skipped, or the caller reads
`forcefield.itp`.

## 2026-09-25 — `OplsXmlReader` drops content silently (routed `/mol:fix`)

Found by opls-gromacs-01 (not fixed there; `OplsXmlReader` now refuses only
non-representable RB rows):

- `ff/forcefield/readers/opls.rs:143-149` skips `<Residues>`, `<ImproperTorsionForce>`,
  `<PeriodicImproperForce>` and every `<Custom*Force>` without a word, so impropers and
  custom terms in the pack vanish;
- `:460-463` drops `<Improper>` children of `<PeriodicTorsionForce>`;
- `:349-360` (`ensure_class_wildcards`, from `:283`) invents placeholder atom types
  (`type_="*"`, `class_=<class>`, empty params) for class-only bonded endpoints.

Each should be an `Err` naming the element, or a modelled style.

## 2026-09-25 — `read_top` reads both branches of `#ifdef` / `#else` (routed `/mol:fix`)

`io/data/top.rs:271-274` skips every `#` line, so both branches of an
`#ifdef` / `#else` block (e.g. `FLEXIBLE` water) are read as topology. The GROMACS
force-field reader evaluates `#define` / `#ifdef` / `#ifndef` / `#else` / `#endif`
since opls-gromacs-01, but io and ff may not share code (architecture-rules.md), so
`io::data::top` needs its own conditional evaluation.

## 2026-09-25 — private unit-constant copies beside `molrs::units` (routed `/mol:refactor`)

`KJ_PER_KCAL` / `NM_TO_ANGSTROM` are private `const` copies in
`ff/forcefield/readers/gromacs.rs:92-93`, `ff/forcefield/writers/gromacs.rs:75-76`,
`ff/forcefield/readers/opls.rs:59-61` and `ff/forcefield/writers/xml.rs:26`, beside the
unit system in `molrs::units` (`core/units/`). One source for each conversion factor.

## 2026-09-25 — OPLS improper assignment (routed `/mol:spec`)

GROMACS OPLS-AA applies its six improper macros (`improper_Z_N_X_Y`, …, defined in
`ffbonded.itp`) through `.rtp` residue entries, not through `[ dihedraltypes ]`. The
GROMACS force-field reader records `#define` names but never expands bodies, so OPLS
impropers are not assigned by molrs typing. Owed as its own spec.

## 2026-09-25 — remaining Rust convenience free functions (routed `/mol:refactor`)

opls-gromacs-01 deleted `read_gromacs_top_ff` and `write_gromacs_top_ff(_str)`. The
same pattern — a free function wrapping a reader/writer type — remains for
`read_amber_prmtop_ff` (`ff/forcefield/readers/prmtop.rs:75`), `read_forcefield_xml(_str)`
(`ff/forcefield/xml.rs:46`, `:59`) and `write_forcefield_xml(_str)`
(`ff/forcefield/writers/xml.rs:342`, `:348`), re-exported at `ff/mod.rs:13-26`.
Callers should construct the reader/writer (CLAUDE.md § Prefer); binder callers migrate
with them.

## 2026-09-25 — prmtop impropers still keep the first parameter set per name (routed `/mol:fix`)

`readers/prmtop.rs` (~:544, `terms.values().next()`) reduces two improper rows that
share a name but carry different parameter sets to one, silently. Whether to reject or
sum is unspecified; bonds/angles/dihedrals already refuse (system-forcefield-02).

## 2026-09-25 — defining a type scans its style linearly

The single insert path checks the conflict rule by a linear scan of the style's types,
so building a table is O(n²) in its row count (debug build: OPLS 0.85 s, GAFF+GAFF2
0.98 s, MMFF 0.66 s, UFF 0.42 s, including ~0.4 s cargo overhead). A name index on
`Style` is the follow-up if a larger table or a hot path appears.


## 2026-09-25 — OPLS strict typing returned Ok for partly typed molecules (fixed same day)

Strict mode (`NoMatch::Error`) fails only for a bonded term whose endpoints are all
typed and that no bonded type matches. Atoms no def matches stay untyped
(`typifier/opls/typing.rs:242-243`: "strict-mode failure is the consumer's policy" — no
consumer applies it), and `typifier/opls/assign.rs` skips every bond/angle/dihedral with
an untyped endpoint before looking at the policy (`:355-357`, `:377-380`, `:397-404`).
`typify_labeled_graph` (`opls/mod.rs:163-170`) never checks coverage. Seen by the
system-forcefield-04 A/B harness: strict typing of 1,3-butadiene, caffeine and
N-methylacetamide is `Ok`, then compiling fails (`bonds block missing "type" column` /
`unknown bond type ''`). Fixed: strict `typify_labeled_graph` returns `Err` naming every
untyped atom (`atom {i} ({element})`), and the bonded pass refuses an untyped endpoint
under `NoMatch::Error`; tests in `opls/mod.rs` and `opls/assign.rs`. (The bond-order gap that made strict
OPLS refuse every C=C / C=O molecule was closed by opls-gromacs-03.) system-forcefield-07's
`Match::write_onto` checks shape, not coverage, so it does not catch this; 07's OPLS
`r#match` must carry the fix, not the skip.

## 2026-09-25 — OPLS defs assumed bond-order-agnostic matching (resolved by opls-gromacs-03)

The embedded OPLS defs (`ff/params/oplsaa.rs`, same text as molpy's
`data/forcefield/oplsaa.xml`, from foyer) write neighbours as `[C;X3](C)(H)H`
(opls_143), `[C;X3]([O;X1])[N;X3]` (opls_235): foyer's graph matching ignores bond order.
In molrs SMARTS an unmarked bond is single-or-aromatic (`perceive/smarts/ast.rs:290-291`,
`:322`), so every def whose pattern crosses a `=` bond never matches: alkene and carbonyl
carbons stay untyped, and the miss cascades (N-methylacetamide's amide N becomes amine
`opls_901`). Fix needs its own spec and A/B (it changes assigned types): compile OPLS defs
with an any-order unmarked bond, or rewrite the defs with `~`/`=`. On the backmap
critical path (the methacrylate monomer has C=C and C=O).

Resolved 2026-09-25 (opls-gromacs chain): the rules are molrs-owned Daylight SMARTS in
`ff/params/oplsaa_typing.rs` (explicit bonds, `[#1]` hydrogens, aromatic case), matched after
aromaticity is perceived on a private copy; opls_150/178 added; ranking is pairwise dominance.

## 2026-09-25 — the `ParamSource` bidirectional gate test does not exist (routed `/mol:fix`)

`ff/potential/registry.rs` cited `tests/ff/potential/param_source_gate.rs` and CLAUDE.md
§ Potential System said "a bidirectional gate makes it a test"; no such test exists
anywhere. system-forcefield-04 corrected both texts. Owed: a unit test in
`ff/potential/registry.rs` asserting, per registered kernel, that a `TypeRows` kernel
reads its type rows and a `PerInstance` kernel reads frame columns only.

## 2026-09-25 — data-file coefficients refuse explicit cross pairs (routed `/mol:spec`)

Since system-forcefield-06, `LammpsFfWriter::write_data_coeffs_str` refuses a force
field holding an explicit cross pair between two used atom types ("a data-file Pair
Coeffs section holds self pairs only"); before, it dropped the pair silently. The
`*.ff` include writes it. The complete answer is a `PairIJ Coeffs` section in the data
file when explicit cross pairs exist; owed as its own spec.

## 2026-09-25 — type rows on `PerInstance` styles are export and conflict records

After system-forcefield-07 every typifier defines, in its output force field, one type
per stamped name. For MMFF and UFF (`ParamSource::PerInstance`) the kernels read the
per-instance parameters stamped on the frame; the type rows under those styles are never
compiled. They exist so the output names every parameter set used (export, merge) and so
two terms that share a name with different parameters are a `TypeConflict`. A label must
therefore carry every input that changes the parameters (MMFF `stbn_type` is
`{sbt}_{i}_{j}_{k}` in the angle's own node order; UFF labels carry bond orders after
`@`).

## 2026-09-25 — typing is `Typing<T>`; typifiers only match

`Typifier` requires exactly `r#match(&self, &mut Atomistic) -> Result<Match, String>` and
`library(&self) -> &ForceField`. `Typing<T>` owns the typifier and the output force
field (seeded by `library().empty_like()`); `typify(&mut self, &Atomistic)` matches a
copy and runs `Match::write_onto` (validate → stamp → define). No typifier overrides
typing. UFF bond/angle parameters are computed on the label's canonical orientation, so
a reversed term may differ from 0.14 by ≤1 ulp.

## 2026-09-25 — the estimator reads all-caps OPLS classes as element symbols (routed `/mol:debug`, same chain)

`ff/typifier/estimate/mod.rs:396-403` (`element_from_token`) treats a token whose second
letter is upper case as a "title-cased" element symbol and `Element::from_str` accepts it
case-insensitively, so OPLS classes `CA CM CN CO CR CS CU NA NB NO OS HO HS` resolve to
Ca/Cm/…/Hs and are returned in their raw spelling (never equal to a mass-derived
symbol). Row classes never sit in the mass map, so the estimator refuses every
element-compatible substitution at those slots (117/300 bond, 492/932 angle,
510/1048 dihedral OPLS rows involved): worse analogs, empirical fallbacks, and
dihedrals dropping to the `no_torsion` placeholder. GAFF (lower-case types, mass-derived
elements) is unaffected. Fix: element of a class from its member types' mass, with a
corrected true-title-case token fallback that returns `Element::symbol()`.

## 2026-09-25 — typifier binding surfaces (system-forcefield-09)

- Rust `Typing<XTypifier>` ↔ Python/wasm `XTypifier`: the binding class holds the
  `Typing` (Python: `Typing<Box<dyn Typifier + Send + Sync>>`; a Python subclass holds
  its own output `ForceField`, seeded by `library().empty_like()`, and runs the Rust
  `Match::write_onto`). A Python subclass that defines `typify` is a `TypeError` at class
  creation; native classes are construct-only.
- wasm exposes no `forcefield()` / `library()` (documented no-FF-handle surface);
  `toPotentials` compiles the typing output, so it must follow `typify`.
- Class-set asymmetry: Python binds MMFF94, MMFF94S, OPLSAA, ATD (not UFF); wasm binds
  UFF, MMFF94, MMFF94S (not OPLSAA, ATD).
- Routed to the molpy chain: `molpy/src/molpy/typifier/base.py:99` defines `typify`
  (now a `TypeError` against molrs 0.15), and `base.py:115` passes the caller's graph,
  not a copy, to `match`, so molpy typing mutates its input.

## 2026-09-25 — mixing mandatory on every `lj/cut` (routed `/mol:spec`)

An `lj/cut` style that declares no `mixing` is evaluated under `Mixing::UNDECLARED`
(arithmetic): `ff/potential/pair/lj_cut.rs:643-646` (`pair_lj_cut_ctor`) and `:714-717`
(`pair_lj_cut_typed_ctor`). opls-gromacs-02 made the LAMMPS writer name that rule
(`pair_modify mix arithmetic`), so an export no longer mixes geometrically behind the
kernel's back — a patch at the seam. The principled fix is the `coul/cut` one: every
`lj/cut` must declare `mixing`, and a style without it is an `Err`, never a silent
default. Owed as its own spec. Still open meanwhile: the writer's hybrid branch
(`ff/forcefield/writers/lammps.rs:616-618`) writes no rule at all, so an undeclared `lj/cut` inside a `pair_style hybrid`
still mixes geometrically in LAMMPS.

## 2026-09-25 — OPLS engine unit tests still write foyer-style defs (routed `/mol:refactor`)

opls-gromacs-03 rewrote the shipped rules (`ff/params/oplsaa_typing.rs`) as Daylight
SMARTS, but the older hand-written fixtures in `ff/typifier/opls/layered.rs`,
`typing.rs` and `deps.rs` tests (and `opls/mod.rs`'s estimator fixtures) still write
foyer-dialect defs: unmarked bonds, a bare `H` as a hydrogen atom
(`[C;X4](C)(H)(H)H`, `H[C;X4]`, `[O;X2](H)([!H])`), uppercase `C` for ring carbons.
They pass because each fixture only crosses single bonds and never meets an aromatic
atom, so an unmarked bond and a bare `H` happen to read right — but they document a
dialect the shipped table no longer uses, and a copy into a real rule set would
reintroduce the defect this chain fixed. Owed: rewrite them to the header conventions of
`oplsaa_typing.rs` (`-`/`=`/`#`/`:`, `[#1]`, lowercase aromatics), assertions unchanged.

## 2026-09-26 — `molrs::op` is functional by operator ruling (scoped exception)

**Decision:** `molrs::op` exposes plain value types (`Rigid`, `Fit`,
`Freedom`) and free functions — vector arithmetic, 3×3/4×4 linear algebra,
rigid and quaternion kernels, weighted superposition, centroid, uniform S²
directions. The CLAUDE.md OOP default does not apply inside `op`, and only
there.
**Why:** grill 2026-09-26 (assembly chain). These are pure numeric kernels with
no natural owner; nine private vector copies, four quaternion copies, three
centroid copies and two Horn kernels existed because there was no shared base.
**Status:** locked (scope: `molrs::op`)

## 2026-09-26 — `core::types` re-exports the `op` array aliases permanently

**Decision:** `F`, `F3`, `F3x3`, `FN`, `FNx3`, `F3View`, `FNx3View` live in
`op::types`; `core/types.rs` re-exports them, and `molrs::types::*` stays the
canonical crate-root spelling for downstream code. New code inside
`molrs/src` imports `crate::op::types` directly. The re-export is never
widened to the stack aliases `Vec3`, `Mat3`, `Quat` — those have exactly one
path, `op::types`.
**Why:** downstream (molpack, binders) spells `molrs::types::F`; moving the
path would break every consumer for no gain.
**Status:** locked


## 2026-09-26 — routed to `/mol:refactor` (found by assembly-01)

- The `core::math` special functions (`complex`, `spherical_harmonics`,
  `wigner3j`, `wigner_d`) are pure numerics with no assembly role; moving them
  into `op` is a separate refactor.
- The graph-transform systems `translate`, `scale`, `rotate` in
  `core/spatial/geometry.rs` are free functions
  taking `&mut MolGraph`; the OOP shape is a `MolGraph::transform` method.
**Status:** open

## 2026-09-26 — `scale_lj::center_of_mass` passes non-finite coordinates through (routed `/mol:fix`)

**Found by:** assembly-01 (the op redirect). A non-finite *mass* is now refused
(`ScaleLjError::InvalidMass`); a NaN/inf *coordinate* still yields a NaN
fragment centre that flows into the scaled LJ parameters. Fix: validate the
coordinates the way `op::superpose` does (`NonFinite { index }`) and refuse.
It is also a second mass-weighted-centre entry point beside `geometry::center`
(backmap-primitives-01), not reused because its input is `FragmentAtoms` and it
weights a non-positive mass as 1 where `center` refuses a negative mass
(`CenterError::BadMass`) and weights a zero mass as 0; the fix should settle
whether that policy difference is intended.
**Status:** open

## 2026-09-26 — a second Jacobi eigensolver with absolute tolerances (routed `/mol:refactor`)

**Found by:** assembly-01 architect review. `conformer/etkdg/embed4d.rs:167-236`
`jacobi_eigen` is an N×N cyclic Jacobi with absolute tolerances (`off < 1e-30`,
`|apq| < 1e-300`) — the scale dependence `op::linalg` removed for 3×3/4×4
(1e-15·‖A‖_F). Fix: generalise `op::linalg`'s const-generic Jacobi to N×N with
the relative tolerance and delete this copy.
**Status:** open


## 2026-09-26 — one typed dimension per column

**Decision:** `ColumnSpec.dimension: ColumnDim` (`NotAQuantity` /
`Dimensionless` / `Of(PresetDim)` / `Product(PresetDim, PresetDim)`) is the
single truth for what a column measures. The free-text `ColumnSpec.unit` string
is removed. The schema document derives its displayed unit from the dimension
in `real` units (`ColumnDim::unit_in(&UnitPreset::real())`); molrs converts no
frame columns itself — the per-column frame conversion was retired by
backmap-primitives-02, and callers convert one unit at a time.
`PresetDim` replaces the `PRESET_DIMENSIONS` string array, so there is one list
of dimension names. **Why:** the `unit` string had drifted (`"amu"`, `"e"`,
`""` on `x`) and, being free text, could not drive a conversion.
The one undeclared-in-spec Float key, `q0`, was declared `Of(Charge)` (superseded: `q0` removed by assembly-06, port-only connection).
**Source:** assembly-02-io-units §Design 5.
**Status:** adopted

## 2026-09-26 — per-key unit doc strings vs the derived unit column (routed `/mol:docs`)

The `doc` strings on `x` and `vx` still say "Unit follows the force field …;
molrs stores raw numbers", while the schema document now shows a derived
`unit (real)` column (`angstrom`, `angstrom / femtosecond`). Reconcile the
wording with the typed dimension. Routed `/mol:docs`.
**Status:** open

## 2026-09-26 — the `CoarseGrain` frame block `members` is outside the schema vocabulary (routed `/mol:spec`)

**Found by:** assembly-02 architect review; narrowed by backmap-primitives-05,
which removed the CG-bond block. `members` (`ibead`, `atom`, both UInt) is in
neither `SCHEMA_BLOCKS` nor `keys.rs`, so the schema document does not describe
it. It is not a relation block — `ibead` indexes `atoms` rows but is not a
schema endpoint — so `Frame::subset` refuses a frame carrying it, whatever the
target block, until it is in the schema (otherwise its rows would go stale when
bead rows are dropped). Declaring `atom` as a schema column would bind its dtype
wherever the key appears, so the right shape is a decision for the
schema-vocabulary spec (the same spec that owes `frag_id`, notes.md 2026-09-21).
**Status:** open

## 2026-09-26 — `Fragment::inherit_frag_ids` ignores a `set_frag_id` error (routed `/mol:fix`)

**Found by:** assembly-03. `fragment.rs` `inherit_frag_ids` does
`if self.set_frag_id(atom, id).is_ok()` and drops the error. It cannot fail
today (live handle, `frag_id` is i32 whenever a label exists), but surfacing it
changes the public `-> usize` signature, which assembly-03 ruled out.
**Status:** open

## 2026-09-26 — the LAMMPS force-field reader drops label-led Coeffs rows (routed `/mol:fix`)

**Found by:** assembly-02. `ff/forcefield/readers/lammps.rs:296-302` ends a
`… Coeffs` section on any line starting with an uppercase letter, so a
type-labelled row such as `CA 0.1 3.4` in `lammps_coeffs_text` is silently
dropped. The data reader now keeps such rows (assembly-02 closed-vocabulary
headers); this parser needs the same header rule.
**Status:** open

## 2026-09-26 — connection is by port only; `SiteMap` / `SITE` / `Q0` retired

**Decision:** two units join only through a port pair (`Port::accepts`,
`Fragment::link` — backmap-primitives-03, which folds the removed handle
branch's charge onto the anchor). `builder/sites.rs` (`SiteMap`, `SiteError`, `SITE_KEY`,
`PRE_REACTION_CHARGE_KEY`) and the schema keys `site` / `q0` are deleted;
`FRAME_VOCAB_VERSION` is 2. Cross-repo consumers follow in their own chains:
molpy `builder/__init__.py`, `builder/assembly/__init__.py`,
`builder/assembly/_polymer.py`, `core/atomistic.py`, `core/cg.py`,
`core/fields.py`; molpack `pack_peo_*`.
The SMIRKS `Reaction` engine stays (consumers: `lib.rs`, `perceive/smarts`,
Python `PyReaction`, molpy `GraphAssembler` / ambertools builders, molpack
examples) and so does `%label` (OPLS typing).
**Why:** two connection mechanisms (SITE maps vs ports) and two grouping keys
(`res_id` vs `frag_id`) described the same join.
**Status:** locked

## 2026-09-26 — `perceive_aromaticity` drops write errors with `let _ =` (routed `/mol:fix`)

**Found by:** assembly-06. `perceive/aromaticity.rs:~764,773,778` discard
`set_atom` / bond-class write errors. At :764 an existing F64 `is_aromatic`
column refuses the Int write, the error is swallowed, and a stale aromatic flag
stays (readers accept F64, `perceive/smarts/ast.rs:76`). The fix needs either a
`Result` return (public signature change) or a coerced write.
**Status:** open

## 2026-09-26 — six multi-frame Python writers still deep-copy their frames (routed `/mol:refactor`)

**Found by:** assembly-07. The nine single-frame writers in
`molrs-python/src/io/mod.rs` borrow through `PyFrame::with_frame`; the six
multi-frame writers still `clone_core_frame()` every frame of the list, because
`with_frame` borrows one store entry at a time: `write_pdb_trajectory` (~:1619),
`write_lammps_dump` (~:1677), `write_lammps_dump_local` (~:1693), `write_dcd`
(~:1720), `write_trr` (~:2047), `write_xtc` (~:2068) (line numbers as of
2026-09-26). Fix: a multi-entry borrow over the frame store.
**Status:** open

## 2026-09-26 — Rust debug ids inside crate-built error strings (routed `/mol:fix`)

**Found by:** assembly-07; retargeted by backmap-primitives-08 after 02 deleted
the builder sites. The binder renders node / port ids as the integer handles
Python sees for every error variant that *carries* an id. One crate refusal
formats the id into text instead, so Python still sees `NodeId(..)`:
`Atomistic::try_from_molgraph` (`core/system/atomistic.rs:~656`, "node {:?}
missing '{}' property"), reachable from Python through `Fragment.to_atomistic`.
The same format sits in the sibling leaf checks `Fragment::try_from_molgraph`
(`fragment.rs:~329`) and `CoarseGrain::try_from_molgraph`
(`coarsegrain.rs:~433`). Fix: a structured id field on that refusal, rendered
by the binder. The `Fragment.link` path (`LinkError::Port` through
`MolGraph::get_relation` / `Fragment::port` / `missing_prop`) is being fixed in
this chain (backmap-primitives-07 amendment 7); the entry stays open for
`try_from_molgraph` and the remaining sites.
**Status:** open

## 2026-09-26 — `Potentials.eval_any` deep-copies its frame (routed `/mol:refactor`)

**Found by:** assembly-07. `molrs-python/src/ff/mod.rs:~786` still
`clone_core_frame()`s: borrowing through `with_frame` would hold the frame store
while it calls Python-implemented potentials that may touch the same store.
Restructure: compile inside `with_frame`, evaluate outside.
**Status:** open

## 2026-09-26 — `op::linalg::inv3` of a matrix whose determinant overflows (routed `/mol:fix`)

**Found at:** the assembly chain-end gate. For finite entries so large that
`det` overflows to +inf while the relative threshold stays finite, `inv3`
returns an all-zero matrix instead of `None` or a scaled inverse. Behaviour
unchanged by the gate's lint fix; decide `None` (overflow = unrepresentable) or
compute on a rescaled matrix.
**Status:** open

## 2026-09-26 — backmap is primitives the caller composes (supersedes the assembly builder)

**Decision:** molrs ships no assembly engine. Backmapping means replacing each
group of coarse-grained beads with the all-atom molecule it stands for; the
caller composes it from these primitives:

- `perceive::SubgraphMatcher::new(&pattern).find(&target)` finds bead groups
  (whole molecule ↔ bead group); `SubgraphMatcher` is also re-exported at the
  crate root.
- `CoarseGrain::center(group)`, `Atomistic::center()` and `Fragment::center()`
  delegate to `geometry::center` (`CenterError`; centre of mass for atoms,
  bead-mass-weighted centre for beads).
- `geometry::translate` places, translation only.
- `Fragment::merge` returns (atom map, port map).
- `Fragment::link(a, b)` is the only join.
- `Frame::subset(block, rows)` plus single-molecule `CoarseGrain::from_frame`
  select a molecule.
- `CGSmilesIR` converts only to `MolGraph` leaves (`to_atomistic`,
  `to_fragment`, `to_coarsegrain`).
- A `MolGraph` holds no box: the caller unwraps with `SimBox::unwrap` (Python
  `Box.unwrap`), converts LJ lengths one unit at a time, and wraps with
  `SimBox::wrap` (Python `Box.wrap`).

**Retired with no replacement** (backmap-primitives-02 and -05):

- `FragLibrary`, `Mapping`, `FragGraph`, `Placer` / `TracePlacer`, the
  orienters, `Reacter` / `PortReacter` / `link_many` / `PairError`,
  `Finalizer`, `Assembler` (Rust and Python, including the adaptors and Python
  `Trace`);
- `Frame::convert_units` and `Unconvertible`, `CoarseGrain::from_atom_frame`,
  `CGSmilesIR::to_template` / `to_frag_graph`;
- `op::rigid::{compose, alignment}`, `op::so3::{random_rotations,
  random_angles, rotation_from_uniform}`, `op::superpose::superpose_many`, and
  `op::vec3::perpendicular` (its last caller was `rigid::alignment`; no sibling
  repo consumes it; 02 removed 147 tests in all, not the 143 first counted);
- the `bead` key and the `beads` / `cgbonds` blocks.

Renamed, not retired: `ReactError` became `core::system::link::LinkError`
(backmap-primitives-03).

**Why:** operator rulings and grill decisions of 2026-09-26:

- whole molecule ↔ bead group;
- primitives only, the user composes;
- IR → `MolGraph` leaves only;
- no in-crate frame unit conversion;
- single-molecule `from_frame` + `Frame::subset`;
- translation only;
- core = data, and queries do not sit on data types;
- joining only through port-bearing `Fragment`s;
- no box in `MolGraph`;
- finding bead groups is the matcher's job;
- centre = centre of mass / bead-mass-weighted centre.

**Supersedes** the Assembler, Finalizer and two-doors entries. **Resolves** the
`Mapping.labels`, assembly-06 test gap and assemble→finalize copy entries
(Python now has the move-based `Fragment.to_atomistic()`). The composition
itself lives in molpy's `backmap-` chain.

**Out-of-repo consumers** of the retired vocabulary (backmap-primitives-05
amendment 4):

- molpy `src/molpy/core/cg.py` (`CoarseGrain.to_frame` documents the retired
  blocks, and its `bead_fields` filter checks `"beads" in frame`, so it is now a
  silent no-op) and `tests/test_core/test_cg.py` (asserts the retired blocks and
  `ibead` / `jbead`) — routed to molpy's `backmap-` chain;
- molexp `src/molexp/harness/prompts/workflow_source.py` and its copy molab
  `src/molab/harness/prompts/workflow_source.py` (agent prompts that teach
  `src["beads"]`) — routed to a molexp/molab fix.

**Status:** locked

## 2026-09-26 — known limits of the backmap primitives (recorded, accepted)

- **Placement is translation only:** every copy keeps its orientation, and
  anchor–anchor distances are uncontrolled. The operator accepted relaxation
  afterwards.
- **`find` does not partition:** it returns every induced match, so bead
  pattern 1-1-1-4 in target 1-1-1-4-1-1-1-4 gives
  `[[0,1,2,3],[4,5,6,7],[6,5,4,3]]`. `[6,5,4,3]` overlaps `[0,1,2,3]` on bead 3
  and `[4,5,6,7]` on beads 4–6; choosing a partition is the caller's job.
- **Edge labels are ignored** by the matcher.
- **`to_coarsegrain`** reads `levels[0]` only, writes no coordinates, and does
  not record CG edge order.
- **`Frame::subset`** refuses a frame carrying `members`, whatever the target
  block, and refuses an endpoint column that is not UInt.
- **Round trip:** `from_frame(to_frame(cg))` fails for a multi-molecule
  `CoarseGrain` (`from_frame` refuses more than one `mol_id`).
- **Python row errors split by sign:** `_row_indices`
  (`molrs-python/python/molrs/frame.py`) raises `IndexError` in Python for a
  row below `-n`, while a row at or above `n` is a `ValueError` raised in Rust
  (`Frame::subset`). One error class per fault is open (routed `/mol:fix`).

**Status:** recorded

## 2026-09-26 — `perceive::subgraph` reaches `graph_hash` through `pub(crate)`; graph-hash queries stay in `core` (routed `/mol:spec`)

- **Scope of the rule** (backmap-primitives-01 §0): a computation's home is a
  free function — `core::spatial` for geometric reductions and transforms,
  `perceive` for graph searches — and delegating leaf methods (e.g.
  `CoarseGrain::center`) are allowed where the operator's script uses them.
- `GraphView`, `adjacency_map` and `feasible` in `core/system/graph_hash.rs`
  stay `pub(crate)`; `perceive` → `core` is the allowed direction.
  `node_label_str` is private.
- `SubgraphMatcher` reads the `pub(crate)` `GraphView` and zeroes its packed
  edge-label bits, and `GraphView::build` computes Weisfeiler–Lehman colours
  (iterated neighbourhood hashes) that `find` discards. Open: core gets one
  owner for an unlabelled adjacency snapshot.
- Two notions of a CG match: `SubgraphMatcher` compares `bead_type` only and
  ignores bond order — ruling: a CG bond has no order — while
  `CoarseGrain::is_isomorphic` (through `graph_hash`) prefers `element`,
  compares bond bits and colours by charge. Aligning `graph_hash`'s CG path is
  routed `/mol:spec`.
- Open: `structural_hash`, `canonical_order` and `is_isomorphic` are graph
  queries implemented in `core` rather than in `perceive`. Moving them changes
  public paths and needs its own spec.

**Status:** open

## 2026-09-26 — `test_backmap_seam.py` chains stages by operator ruling (scoped exception)

`molrs-python/tests/test_backmap_seam.py` runs find → center → translate →
merge → link → to_atomistic on hand-built fixtures and asserts seam facts and
hand counts only. It is the one exception to "no multi-stage pipelines"
(CLAUDE.md § Testing Rules), because it is the operator's expressibility
criterion and there is no `regressions/` tree.
**Status:** locked (scope: that file)

## 2026-09-26 — `MolGraph::merge` drops a failed relation and panics on a kind-arity conflict (routed `/mol:fix`)

- `molgraph.rs:1222` (`molgraph.rs:1235` at `b6418561`):
  `if let Ok(rid) = self.add_relation(…)` silently drops a refused relation.
- `molgraph.rs:687`: `register_kind` panics on an arity conflict
  (`assert_eq!`).
- Both are reachable through `MolGraph::merge`. The kind-arity panic is also
  reachable through `Fragment::merge` (two fragments registering one foreign
  kind at different arities through `DerefMut`), and `Fragment::merge` inherits
  the non-atomic `Err`: there is no rollback, so on failure `self` keeps the
  atoms and ports copied so far and the caller gets no atom map.
- The Python `merge` (`Fragment.merge`, and `Atomistic.merge` /
  `CoarseGrain.merge` in the same shape, `molrs-python/src/core/system/molgraph.rs`)
  takes `other` with `std::mem::take` before the core merge validates, so a
  refused merge destroys the argument; `f.merge(f)` raises PyO3's borrow
  `RuntimeError`, not `ValueError`.
- Fix: return the error, register through `try_register_kind`, and validate
  every property against `self` before the first write (or roll back on error);
  `Fragment::merge` then gains atomicity without a change of its own. In the
  binder, validate before consuming `other`, or hand `other` back in the `Err`.

**Status:** open

## 2026-09-26 — Python binder debts found by backmap-primitives-07 (routed `/mol:fix`)

- Open: method-level stub parity — `molrs-python/tests/test_stub_parity.py`
  compares only class names, so a method missing from `_lib.pyi` passes.
- Fixed in 07 (recorded, not routed): the `PyFragment` `KeyError` docstrings
  now say `ValueError`; `PyPerceive` is `module = "molrs.perceive"`.

**Status:** open (stub parity only)

## 2026-09-26 — `CoarseGrain` `DerefMut` can remove `bead_type` (routed `/mol:refactor`)

`core/system/coarsegrain.rs` implements `DerefMut` to the inner `MolGraph`,
which lets a caller delete a bead's `bead_type` and break the
every-bead-has-a-type rule; `SubgraphMatcher` documents the result (such a bead
reads as `""`, `perceive/subgraph.rs` module rustdoc). Fix: stop `DerefMut`
from removing `bead_type`, then drop that sentence from the `subgraph.rs`
rustdoc.
**Status:** open

## 2026-09-26 — store debts found by backmap-primitives-05 (routed `/mol:fix`)

- `Block::get_mut` (`core/store/block/mod.rs`) returns `&mut Column`, so a
  caller can swap in any dtype past `check_schema`. `from_frame` and `subset`
  therefore refuse wrong-dtype `mol_id` / type / endpoint columns with
  `Validation` rather than `expect`. Fix: narrow `get_mut` to typed in-place
  access, or re-run `check_schema` on replacement.
- `EndpointSpec.target` has one value (`"atoms"`) since the `beads` spec was
  deleted. It is kept on purpose for a future node table; collapse it if none
  arrives.
- `io/data/lammps_molecule.rs` still has `get_int("mol_id")` branches (`:520`,
  `:904`), the unreachable-Int pattern 05 removed from `from_frame`.

**Status:** open

## 2026-09-26 — two `CgBuild` constructors (routed `/mol:refactor`)

`cg_build` (`io/smiles/cgsmiles/to_fragment.rs`) passes `""` as the input text,
while `resolve.rs` (two sites) and `instantiate.rs` build
`SmilesErrorKind::CgBuild` directly with the input. `to_coarsegrain.rs` (and
`to_atomistic.rs`) import a sibling conversion file just to build errors. Fix:
move `cg_build` to `cgsmiles/mod.rs` or `error.rs` with an optional input and
route all sites through it.
**Status:** open
