---
title: Generic conformer embedding over element-bearing graphs (ElementGraph)
slug: cgsmiles-02c-conformer-fragment
status: approved
created: 2026-09-21
revised: 2026-09-21
chain: cgsmiles (01a → 01b → 01c → 01d → 01e → 02a → 02b → 02c → 02d → 03)
depends_on: cgsmiles-02a-fragment-core
---

# Generic conformer embedding over element-bearing graphs (ElementGraph)

## Summary

`Conformer::generate` today takes and returns `Atomistic`, so a caller holding a `Fragment` (the 02a newtype carrying `ports` and `frag_id`) must unwrap to a `MolGraph`, embed, and re-promote — losing the typed world on the way out. This spec makes the one existing entry point generic over a new local trait `molrs::conformer::ElementGraph`, implemented by `Atomistic` and `Fragment`, so the caller gets back the same type it handed in, ports and fragment labels intact, with no second public method name. The ETKDG pipeline itself is untouched: it still runs on `Atomistic`, and the generic wrapper only converts at the two ends. Relabelling the hydrogens the pipeline adds is the caller's visible second step (`Fragment::inherit_frag_ids`, a 02a primitive), shown in the public example rather than hidden behind the embed. **Hard prerequisite:** this spec must not merge against `Atomistic::try_from_molgraph` as it stands at `molrs/src/core/system/atomistic.rs:606` — see Design.

## Domain basis

The embedding stage is unchanged ETKDGv3: distance-geometry bounds from experimental torsion preferences, 4D embedding, torsion refinement, MMFF94 cleanup, chiral-volume check.

- Riniker, S.; Landrum, G. A. *Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation.* J. Chem. Inf. Model. **2015**, 55, 2562–2574. DOI: `10.1021/acs.jcim.5b00654`. Implemented in `molrs/src/conformer/etkdg/` and `molrs/src/conformer/distgeom/`.
- Hydrogen interpretation of unconsumed valence follows the CGsmiles R4.17 rule (unconsumed descriptors are hydrogens at the atomistic level; cgsmiles.readthedocs.io syntax/chirality; reference implementation `resolve.py:431-433` @ 910c9ee) already implemented by `perceive::hydrogens::implicit_h_count` — an unconsumed descriptor on a heavy atom is a hydrogen, not a vacancy. Embedding a fragment with its handles present is embedding a real closed-shell molecule, the input class ETKDG already handles.

Units are the repo standard: coordinates in Å, MMFF94 energies in kcal/mol, angles in radians internally. This spec adds no new numeric quantity; it moves the same coordinates through one extra typed conversion at each end.

## Design

### The trait and where it lives

```rust
// molrs/src/conformer/element_graph.rs
pub trait ElementGraph: Sized {
    fn as_molgraph(&self) -> &MolGraph;
    fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError>;
}
```

Two methods, both genuinely needed by `generate` and genuinely shared. `ElementGraph` is defined in `molrs/src/conformer/element_graph.rs` — the module of its only consumer — and re-exported as `molrs::conformer::ElementGraph`. The impls for `Atomistic` and `Fragment` live in that same file (local trait, so no orphan rule applies) and are pure forwards to the inherent methods those types already own. This adds **no** `core → conformer` edge; the dependency arrow still runs `core → perceive → ff → conformer` as `architecture-rules.md` § Module dependency rules requires. The exact in-repo precedent is `io::reader::FromFrame` (`molrs/src/io/reader.rs:69`): a `Sized` trait with a `Self`-returning constructor, defined in the module of the generic function that consumes it (`FrameReader::read_as<T: FromFrame>`, `reader.rs:57`). Placing `ElementGraph` in `core` was considered and rejected: zero `core` consumers and one in-tree consumer, against `FrameAccess`'s ~75 — a trait with one consumer belongs beside that consumer (CLAUDE.md § Shape check, point 3).

**No post-embed hook.** A `finish_embed` default-no-op method with one overriding implementor was weighed and rejected: it names the *caller's* pipeline stage on a trait about graphs, is silently a no-op for one implementor, and would make `generate` four steps behind one name (CLAUDE.md § Forbid, all-in-one façade; `architecture-rules.md` § Trait design — a capability only some implementors have is not a default method). Composition is the caller's job and is written out in the public example (below).

**Contract, documented on the trait.** `try_from_molgraph` enforces a per-node `element` property; `as_molgraph` borrows the same graph it was built from; the implementors are `Atomistic` and `Fragment` and no one else. `CoarseGrain` has the same inherent methods and deliberately does **not** implement `ElementGraph` — its nodes carry bead types, not elements, and ETKDG's bond-length estimation, ring geometry and force-field selection all read `element`. Compare `perceive::Perceive`, which is graph-in/graph-out on raw `MolGraph` because perception results are properties any wrapper can carry. The conformer pipeline cannot be shaped that way here: the caller must get the *same typed world* back, ports included, and a bare `MolGraph` return would force every `Fragment` caller to re-promote and re-derive what it already knew.

**Trait principle 1 drift, named and routed.** `architecture-rules.md:92` states trait principle 1 absolutely ("no `Self` in return position, no generic methods"). That is already contradicted in-tree by `FromFrame` (`Self`-returning constructor, `io/reader.rs:69`) and `FrameAccess::visit_block<R>` (generic method, `core/store/frame_access.rs:46`), both static-dispatch bounds never used as trait objects; `ElementGraph` is the third. An implementation spec must not rewrite the rule that governs it: the amendment ("object-safe when used as a trait object; static-dispatch bounds may name `Self` or be generic") is recorded through `/mol:note` by the orchestrator as a provisional entry citing the three witnesses, to be promoted into `architecture-rules.md` after this link ships. The implementation does not edit `architecture-rules.md`.

### `Conformer::generate` becomes generic

```rust
pub fn generate<M: ElementGraph>(&self, mol: &M) -> Result<(M, ConformerReport), MolRsError> {
    let work = Atomistic::try_from_molgraph(mol.as_molgraph().clone())?;
    let (out, report) = etkdg::generate_3d_impl(&work, &self.opts)?;
    Ok((M::try_from_molgraph(out.into_inner())?, report))
}
```

One name, one entry point — the `Naming` rule "no dual public names for the same symbol" is satisfied, and no façade is added (CLAUDE.md § Forbid, all-in-one façades): `generate` remains the single named concern it already was. The two existing call sites, `molrs-wasm/src/conformer.rs:85` and `molrs-python/src/conformer/mod.rs:250`, pass `&Atomistic` and infer `M = Atomistic`, so both stay source-compatible and neither binder file is edited by this spec.

**Cost, stated plainly:** the entry path now performs **2 clones of the graph where 1 existed, ahead of the embed** — `mol.as_molgraph().clone()` here, plus the clone `perceive::hydrogens::add_hydrogens` (`hydrogens.rs:50`) already made inside `generate_3d_impl`. Both are ahead of the distance-geometry solve, which dominates; no clone is added inside the retry loop; `into_inner` and both `try_from_molgraph` moves are free.

**Why relations and properties survive.** `generate_3d_impl` clones the input, appends H atoms and H bonds (`hydrogens.rs:99-110`), and otherwise only writes `x`/`y`/`z` (`etkdg/mod.rs:250-254`, `write_coords` at `:399-407`; `place_single_atom` at `:387-395`). No stage removes a node, a relation, or a property, so a `Fragment`'s `ports` rows and `frag_id` props pass through untouched. The `ports` kind also rides through the MMFF staging `Frame` as an **unread block**: `annotate_mmff` clones the graph (`ff/typifier/mmff/frame_builder.rs:114`), `to_frame` emits one block per non-empty kind (`molgraph.rs:983-1011`), and `to_potentials` selects blocks by category name only (`ff/potential/compile.rs:260-271`); a future strict block vocabulary would have to admit `ports`. Handles are real hydrogens in the graph and are distinguished from ETKDG-added hydrogens only by their `ports` membership — which is why the `ports` kind must never be confused with the `bonds` kind. The MMFF staging Frame is built from the **`Atomistic`** work graph through `MolGraph::to_frame`, which 02a's all-or-nothing `frag_id` rule (a `Fragment::to_frame` rule) does not cover, so that staging Frame does carry a zero-filled `frag_id` column for the not-yet-labelled hydrogens; this is harmless only because nothing reads that column back — the output `Fragment` is `work`, never re-read from the Frame (`etkdg/mod.rs:251-254`).

### Hard prerequisite: idempotent kind re-registration (02a)

Today `Atomistic::try_from_molgraph` (`atomistic.rs:606-627`) resolves each standard kind by name and, on a miss, falls back to `KindId(0..3)` **with no arity or name check**. A `Fragment`-derived graph whose first registered kind is `ports` would therefore alias `bonds` onto `ports`: `add_hydrogens` would append H bonds into the `ports` relation, and `conformer/distgeom/torsion_prefs.rs:138-140` would read port rows as bonds (`g.bonds()` → `[b.nodes[0], b.nodes[1]]`). That is silent chemical corruption, not a type error.

02a fixes this by resolving each standard kind by name with an arity check in **both** `Atomistic::try_from_molgraph` and `CoarseGrain::try_from_molgraph` — a conflicting arity returns `MolRsError::validation` (a `Result` constructor never reaches the `register_kind` assert at `molgraph.rs:470-473`) and a matching or absent kind goes through the documented-idempotent `MolGraph::register_kind(name, arity)`. 02a also owns the guard tests, in `atomistic.rs` and `coarsegrain.rs` next to the fix, each promoting a hand-built `MolGraph` whose first registered kind is `ports` **with at least one populated port relation** (an empty kind makes `n_bonds() == 0` vacuous) and asserting `n_bonds() == 0` and that a subsequent `add_bond` lands in a kind distinct from the ports kind. **Do not merge this spec against `atomistic.rs:606` as it stands**; this spec depends on 02a's fix and its tests rather than re-asserting a foreign module's contract.

### `frag_id` on pipeline-created atoms is the caller's visible step

With `add_hydrogens = true` — the default (`options.rs:47`) — `add_hydrogens` appends bare `Atom`s carrying only `element`, `mass`, `x`, `y`, `z` (`hydrogens.rs:99-107`), each with exactly one bond to its heavy parent (`:107-110`). The output `Fragment` therefore contains unlabelled hydrogens. The repair is a `Fragment` invariant ("a `frag_id` propagates to atoms attached to a labelled atom") and lives on 02a's type as `Fragment::inherit_frag_ids(&mut self) -> usize` — **defined, documented and unit-tested in 02a** (this spec extends 02a's surface by naming it; 02a's acceptance carries it). Its contract: one pass over a snapshot of the input labels (never a fixpoint); only degree-1 nodes without a `frag_id` whose single neighbour carries one are labelled; returns the number of atoms labelled; documented precondition "intended for atoms newly attached by a pipeline — a degree-1 atom deliberately left unassigned will be relabelled". `generate` does not call it. The public example composes the two steps:

```rust
let (mut frag, report) = Conformer::new(opts).generate(&frag)?;
let labelled = frag.inherit_frag_ids();
```

**Frame-boundary caveat, stated not hidden.** `MolGraph::to_frame` emits columns through `emit_column` (`molgraph.rs:150-185, 975`), which drops the validity mask and writes the default (`0`) for a null `frag_id`; an unassigned `frag_id` is therefore not representable in a `Frame`. 02a's `Fragment::to_frame` emits the `frag_id` column **only when every atom carries one** (all-or-nothing), so a partially labelled fragment round-trips as "unlabelled" rather than as "labelled 0"; a caller that wants labels in the Frame calls `inherit_frag_ids` first. 02d must respect the same rule at the Python seam.

### Forward shape for 02d (Python), sketched now

pyo3 cannot export a generic method, and `notes.md` § Binding-surface symmetry (2026-08-10) forbids letting the Python surface grow a second name for one Rust concern. 02d will keep exactly one `Conformer.generate(mol)`, which extracts `PyFragment` first and `PyAtomistic` second — leaf-first, the ordering `with_world_mut` uses at `molrs-python/src/core/system/molgraph.rs:1608-1621` — calls this same Rust `generate`, and wraps the result in the class matching the input. This is a **new shape** for that file: the comment at `:1596-1603` says algorithms are module functions taking `PyAtomistic` directly, so 02d proposes `PyAny` dispatch inside a class method and must say so. Two further asymmetries 02d inherits and must name: `molrs-wasm` has no conformer surface at all (`grep Conformer molrs-wasm/src` is empty), so widening the Python surface widens an existing gap; and `PyConformer` is declared `subclass` for `molpy.conformer` (`molrs-python/src/conformer/mod.rs:142-168`), so changing what `generate` accepts and returns is molpy-visible.

### Reuse decision

- `reuse` `Atomistic::try_from_molgraph` / `into_inner` / `as_molgraph` (`atomistic.rs:606-642`) — the `Atomistic` impl of `ElementGraph` forwards to them and adds nothing.
- `reuse` 02a's `Fragment::try_from_molgraph` / `as_molgraph` / `into_inner` — likewise for the `Fragment` impl.
- `reuse` `etkdg::generate_3d_impl` (`etkdg/mod.rs:50`) — called unchanged; no second pipeline.
- `reuse` `MolGraph::register_kind` (`molgraph.rs:464`) — the already-idempotent primitive 02a's fix is built on; this spec adds no new registration helper.
- `reuse` 02a's `Fragment::inherit_frag_ids` — called by the caller, not by `generate`; this spec is its first named consumer.
- `pattern` `io::reader::FromFrame` (`reader.rs:69`) — `ElementGraph`'s shape, placement and doc structure follow it so the new code reads like the existing code.

## Files to create or modify

- `molrs/src/conformer/element_graph.rs` (new) — `ElementGraph` trait, its `Atomistic` and `Fragment` impls, and the round-trip / bound-witness tests.
- `molrs/src/conformer/mod.rs` — declare and re-export `element_graph`; make `Conformer::generate` generic; extend the module `//!` doctest at lines 9–16 with the `Fragment` path (`generate` then `inherit_frag_ids`).
- `molrs/src/perceive/hydrogens.rs` — test only: `add_hydrogens` on a ported graph leaves `n_ports()` unchanged, gives each new H exactly one bond, and yields nine atoms for the C–C–O fixture.
- `CLAUDE.md` — § Conformer Pipeline, line 295: record the generic signature (the current line is falsified by this change).

## Tasks

- [ ] Write failing unit tests for `ElementGraph` in `molrs/src/conformer/element_graph.rs` (`#[cfg(test)]`: `Fragment` → `Atomistic::try_from_molgraph` → `into_inner` → `Fragment::try_from_molgraph` with no ETKDG, asserting ports, per-atom `frag_id`, bond count and port count separately; a compile-time bound witness `fn assert_bound<M: ElementGraph>() {}` instantiated for both types)
- [ ] Write the failing `add_hydrogens` contract test in `molrs/src/perceive/hydrogens.rs` (a hand-built C–C–O graph with two handle hydrogens that are real bonded atoms additionally tagged in a `ports` kind, no `h_count`/`formal_charge` set: after `add_hydrogens`, `n_ports()` is unchanged, every new H has exactly one bond, and the atom count is nine — 3 heavy + 2 handles + 4 added, hand-derived from valence 4+4+2 minus 4 internal minus 2 handles)
- [ ] Implement `ElementGraph` with its `Atomistic` and `Fragment` impls in `molrs/src/conformer/element_graph.rs`, and re-export it from `molrs/src/conformer/mod.rs`
- [ ] Implement the generic `Conformer::generate<M: ElementGraph>` in `molrs/src/conformer/mod.rs` routing through the unchanged `etkdg::generate_3d_impl`
- [ ] Update the `generate` rustdoc and the module-level `//!` doctest in `molrs/src/conformer/mod.rs` to name `ElementGraph`, the ports/`frag_id` pass-through invariant, the unread `ports` block in the MMFF staging Frame, coordinate units (Å), and to show the `Fragment` path as `generate` followed by `inherit_frag_ids`
- [ ] Update `CLAUDE.md` § Conformer Pipeline (line 295) to the generic signature and name the `ElementGraph` implementors; name the trait-principle-1 amendment (witnesses `io/reader.rs:69`, `core/store/frame_access.rs:46`, `ElementGraph`) as a deferred decision in the implementation summary for the orchestrator's `/mol:note`
- [ ] Verify caller source compatibility with `cargo check --manifest-path molrs-wasm/Cargo.toml` and `cargo check --manifest-path molrs-python/Cargo.toml`, leaving both binder sources unmodified
- [ ] Run full check + test suite: `cargo fmt --check`; `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`; `cargo clippy --manifest-path molrs-cxxapi/Cargo.toml --all-targets -- -D warnings`; `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream`; `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream`

## Testing strategy

Unit tests are `#[cfg(test)]` modules next to the code (`testing.md`; there is no `molrs/tests/` tree and no `regressions/` tree in this repo). Fixtures are hand-built graphs — no `io::smiles` needed, no external oracle, no captured numbers. **No test in this spec runs ETKDG**: a run of several stages in sequence is not a unit test (CLAUDE.md § Testing Rules), and the properties the previous draft wanted from a seeded pipeline run decompose into the three unit tests below, each owned by the unit responsible.

1. **Round trip, no ETKDG** (`conformer/element_graph.rs`). Build a small `Fragment` with two ports and `frag_id` set on every atom, go `Fragment → Atomistic::try_from_molgraph → into_inner → Fragment::try_from_molgraph`, and assert separately: port count preserved, per-atom `frag_id` preserved, bond count preserved, and bonds and ports remain distinct kinds. This is the promotion `generate` performs at each end, tested without the embed between them.
2. **`add_hydrogens` contract** (`perceive/hydrogens.rs`). The nine-atom count and the one-bond-per-H property are properties of `add_hydrogens`, so its file is where a regression is caught. `add_hydrogens` takes `&Atomistic` (`hydrogens.rs:48`), so the fixture is an `Atomistic` whose underlying `MolGraph` has a registered 2-ary `ports` kind holding two relations (handles are real bonded H atoms; no `h_count`/`formal_charge`); the assertions are on the `MolGraph`: the row count of the `ports` kind is unchanged (not `Fragment::n_ports()` — no promotion in this test), every added H has exactly one bond, and the atom count is nine (`implicit_h_count` bills every bond at least 1 via `valence_demand`'s `.max(1)`, so the result does not depend on `BondType`; it breaks only for an unbonded port).
3. **`inherit_frag_ids`** — owned and tested by 02a (`core/system/fragment.rs`): a degree-1 unlabelled H with one labelled neighbour inherits; an unlabelled atom with no labelled neighbour stays `None`; an already-labelled atom is unchanged; the return value counts the labelled atoms. This spec does not duplicate that test.
4. **Bound witness** (`conformer/element_graph.rs`). `fn assert_bound<M: ElementGraph>() {}` instantiated for `Atomistic` and `Fragment`; a compile-time check costing no runtime.
5. **Doctest as the compiled public-API example — and an honest coverage statement.** This repo has no `regressions/` tree; its equivalent is the rustdoc example, which is public API that compiles. The module doctest in `conformer/mod.rs:9-16` is fenced ` ```no_run ` today and stays so: `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` type-checks it and never executes it, because a multi-stage embed is not a unit test and does not belong in any gate (CLAUDE.md § Testing Rules). It shows both the `Atomistic` and the `Fragment` call, the `Fragment` arm followed by `let _n = frag.inherit_frag_ids();`. **Consequently no test in this spec's verification executes ETKDG on a `Fragment`**: the ports/`frag_id` survival property rests on `generate_3d_impl` being byte-for-byte unchanged (ac-005 of the review: verified clone-then-write-coords, no node removal) plus the two conversion unit tests (1) and the `add_hydrogens` contract test (2), which together cover every step `generate` performs around the embed. That is a deliberate trade, stated here rather than implied.

Binder surfaces get no new test here: 02d owns the Python seam, and the WASM path is unchanged. The gate proves source compatibility instead, by `cargo check`-ing both binder manifests with their sources untouched. `ElementGraph` and its impls vanish under `--no-default-features` (the `conformer` feature), which is exactly what the wasm/Pyodide checks are there to catch.

## Out of scope

- **The 02a prerequisite itself.** Idempotent `register_kind` in `Atomistic::try_from_molgraph` **and** `CoarseGrain::try_from_molgraph`, their guard tests (with a populated port relation), `Fragment::inherit_frag_ids` and its test, and the all-or-nothing `frag_id` column in `Fragment::to_frame` land in `cgsmiles-02a-fragment-core`. This spec depends on them and must not merge against `atomistic.rs:606` as it stands.
- **Python and WASM binder edits.** `Conformer.generate` accepting a `PyFragment` is 02d (with the three asymmetries named in Design). No binder source changes here — the gate asserts their diffs are empty.
- **`.claude/notes/architecture-rules.md` trait principle 1.** Amended through `/mol:note` (provisional, three witnesses), not by this implementation.
- **`CoarseGrain: ElementGraph`.** Excluded by contract, not by oversight: bead nodes carry no `element`, and ETKDG reads `element` at three stages.
- **Port-aware geometry.** Ports pass through as relations; nothing in this spec constrains a handle's direction, dihedral, or capping geometry during the embed. Alternatives considered and declined for this link: seeding a port-direction restraint into `DgConstraints`, and excluding handle hydrogens from the MMFF cleanup. Both change embedding physics and belong in their own spec with their own validation.
- **Performance work on the added clone.** The +1 clone is recorded, not optimised; a borrow-based path would require `generate_3d_impl` to accept a `&MolGraph` view and is a separate refactor.
- **A strict Frame block vocabulary.** The `ports` block passes through the MMFF staging Frame unread today; declaring it belongs to the pending schema-vocabulary spec.
