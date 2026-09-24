---
title: MetaMap iterates in insertion order
slug: frame-meta-dict-parity-01-ordered
status: in-progress
created: 2026-09-22
chain: frame-meta-dict-parity (01-ordered → 02-binder-order → 03-untyped-write → 04-dict-views → 05-document → 07-sequence)
depends_on:
revised: 2026-09-22
---

# MetaMap iterates in insertion order

## Summary

`MetaMap` is backed by a `HashMap`, so every consumer that iterates frame metadata — the extended-XYZ comment-line writer, the Zarr frame-group attribute writer, the WASM `metaNames()` binding, the Python `FrameMeta.__repr__` — emits keys in an order that changes between runs and between builds. This link replaces the inner container with `IndexMap` so metadata iteration is insertion-ordered and therefore deterministic, replaces both `BTreeMap` meta halves of the serde wire form with `IndexMap` so a frame survives a round trip with that order intact, and removes the owned `IntoIterator for MetaMap` impl whose associated type leaked `std::collections::hash_map::IntoIter` into the published API. It is link 01 of the `frame-meta-dict-parity` chain; the Python `FrameMeta` sort removal and the `sequence.rs` schema maps are later links.

## Domain basis

This link declares no physics: there are no equations, constants or units in scope and no `scientist` input was required. The scientific stake is reproducibility rather than correctness — an extended-XYZ file and a MolRec Zarr store written twice from the same `Frame` currently differ in key order, which defeats byte-level comparison of simulation outputs. No DOI/arXiv reference applies.

## Design

### 1. `MetaMap`'s container — `molrs/src/core/store/meta.rs`

`MetaMap` (`meta.rs:306`) becomes `MetaMap(IndexMap<String, MetaValue>)`. `indexmap` is added to `molrs/Cargo.toml` as an **always-on** dependency, because `core` is always compiled. It costs nothing new in the dependency tree: `indexmap` 2.14.0 is already resolved (`molrs/Cargo.lock:675-677`).

Method-by-method consequences:

- `iter` / `keys` / `values` (`meta.rs:343-351`) become insertion-ordered.
- `remove` (`meta.rs:340`) **must** route to `IndexMap::shift_remove`, not `swap_remove`. `swap_remove` is O(1) but relocates the last key into the removed slot, which destroys the order of the surviving keys and would make the guarantee a lie after the first deletion. The signature does not change, so no caller in any binder moves.
- `extend` (`meta.rs:352`), `insert`, `get`, `get_mut`, `contains_key`, `len`, `is_empty`, `clear`, `with_capacity` keep their current signatures and semantics; re-inserting an existing key keeps that key's original position, which is the behaviour `frame.meta["t"] = frame.meta["t"]` already documents as an identity in `molrs-python/src/core/store/frame.rs:107-112`.
- `#[derive(PartialEq)]` stays correct: `IndexMap`'s `PartialEq` compares as a map, order-independently, so `Frame` equality is unchanged by this link.

### 2. The two `IntoIterator` impls — `molrs/src/core/store/meta.rs`

Both current impls name a private container type in their public associated type (`meta.rs:357-371`), which is why the container could not be swapped without a breaking change either way.

- **Owned `impl IntoIterator for MetaMap` is deleted.** The serde call `frame.meta.extend(r.meta)` is not a consumer: `r.meta` is the `FrameRepr` field. One real consumer did exist: `molrs/src/io/data/cif.rs` `FrameInProgress::build_frame` moved `self.meta` with `for (k, v) in self.meta`. That loop now copies through `iter()` into `extend`, in insertion order. No other owned consumer remains. `architecture-rules.md` ("delete façades, not deprecate") authorizes the deletion.
- **Borrowed `impl<'a> IntoIterator for &'a MetaMap` (`meta.rs:365-371`) is kept**, because it is live: `molrs/src/io/data/xyz.rs:1929` (`for (k, v) in frame.meta_ref()`) and `molrs/src/io/zarr/frame_io.rs:535` (`for (k, v) in &frame.meta`) both use it. Its `type IntoIter` becomes a new named newtype `MetaIter<'a>` wrapping `indexmap::map::Iter<'a, String, MetaValue>`, implementing `Iterator`, `ExactSizeIterator` and `DoubleEndedIterator` by delegation. The newtype exists so a public signature does not name `indexmap`. It is not a compatibility wrapper and it does not promise that a later container swap stays source-compatible. `MetaMap::iter` returns `MetaIter<'_>` for the same reason, replacing its current `impl Iterator<Item = ...>`.

`MetaIter` is a type in `core::store::meta`, owned by the type whose iteration it names; it is not a free helper and adds no second way to spell an existing operation.

### 3. The serde wire form — `molrs/src/serialize.rs`

**Wire clause, stated unambiguously.** The frame wire form carries two map halves in each direction, and this link changes exactly one of the two, in both directions:

- The **blocks** halves stay `BTreeMap` — the serialize side (`serialize.rs:509`) and the `FrameRepr` deserialize side (`serialize.rs:525`). Block names continue to be emitted sorted. This link does not touch them.
- The **meta** halves are **both replaced with `IndexMap`** — the serialize side (`serialize.rs:511-513`) and the `FrameRepr` deserialize side (`serialize.rs:527`). Both halves, both directions, in this link.

Read the resulting behaviour as: a serialized frame carries its block names alphabetically and its meta keys in the insertion order they had on the `Frame`, and decoding reproduces **both** orders. It does *not* mean "meta is insertion-ordered on the way out and sorted on the way back in" — a `BTreeMap` left on the deserialize side would re-sort on decode and silently undo the guarantee on every MessagePack/JSON round trip. Replacing only one half would be worse than replacing neither, which is why the two edits are one task.

After the change, `serialize.rs:546` `frame.meta.extend(r.meta)` drives `IndexMap`'s `IntoIterator` and therefore preserves the decoded order; it stays a non-consumer of `MetaMap`'s own (now deleted) owned impl.

### 4. Behaviour changes this link causes — full disclosure

Three call sites change their observable output order from `HashMap`-arbitrary to insertion. All three are named here; one is contracted, two are recorded as deliberately uncontracted in this link.

- **`molrs/src/io/zarr/frame_io.rs:534-537` — contracted.** `:534` allocates the `serde_json::Map`, `:535` is the loop that fills it from `&frame.meta`. The emitted Zarr frame-group attribute order becomes insertion order. This one **is** contracted here, by a round-trip test in `frame_io.rs`'s own test module, because `write_frame_group` (`frame_io.rs:522`) and `read_frame_group` (`frame_io.rs:631`) jointly own the attribute map: the writer constructs it and the reader re-reads it (`frame_io.rs:640`), so "the order survives a write/read pair" is a property of that pair and of nothing else. The test module already has the fixtures (`store_in`, `TempDir`, `FRAME`, and the `group.attributes()` assertion pattern at `frame_io.rs:1124-1128`).

  *Dependency to verify, not assume:* this claim holds only if `serde_json` is compiled with `preserve_order` (otherwise `serde_json::Map` is a `BTreeMap` and attributes are already sorted). `molrs/Cargo.lock:1471-1481` resolves `serde_json` 1.0.151 **with** `indexmap`, which `serde_json` pulls only under `preserve_order`, so a transitive dependency enables it today. No manifest in this repo names it. The round-trip test is the guard: if the feature is not active for the `full,filesystem,stream` gate set, the test fails loudly, and the fix is to add `serde_json = { version = "1", features = ["preserve_order"] }` to `molrs/Cargo.toml` — a file already in this link's scope.

- **`molrs/src/io/data/xyz.rs:1929` — disclosed, deliberately not contracted here.** `for (k, v) in frame.meta_ref()` builds `comment_parts` (`xyz.rs:1912-1941`), so the `key=value` token order of the comment line of **every extended-XYZ file `write_xyz_frame` (`xyz.rs:1778`) emits** is `HashMap`-arbitrary today and becomes insertion-ordered after this link. That is a user-visible file-format output change and a determinism fix of exactly the same class as the Zarr one, and it is fixed by this link.

  The decision is to **not** add an ordering test in `xyz.rs`, on the same ground that makes the `frame_io.rs` test right. The comment line's token order is *derived*, not owned: `xyz.rs` neither chooses nor normalizes it, it forwards `MetaMap`'s iteration order verbatim into a `join(" ")` (`xyz.rs:1957`). A test there would re-assert `core`'s container guarantee through an I/O writer — a second module's test for another module's behaviour, which is the duplicate that `CLAUDE.md` and `.claude/notes/testing.md` exclude from the suite. The guarantee is contracted once, in `meta.rs`'s own tests. Writing the decision down here is what keeps it from being silent debt; it is not a claim that the change is invisible.

- **`molrs-wasm/src/core/frame.rs:489-497` `metaNames()` — disclosed, deliberately not contracted here, rustdoc unchanged.** The returned array order becomes insertion order. No contract is added and the rustdoc stays silent because nobody can currently depend on the order: the example at `:486` is prefixed `// e.g.` and the test at `:757-763` asserts `len` plus two `contains` calls, never a position.

### 5. Interim window this link opens in the Python binding

After this link and before link 02, `molrs-python`'s `FrameMeta` is deliberately split:

- `__repr__` (`molrs-python/src/core/store/frame.rs:229-231`) goes through `as_dict` (`:163-169`), which iterates `MetaMap` raw, so **repr becomes insertion-ordered**.
- `__iter__` (`:221-223`) is `PyList::new(py, self.keys()?)`, and `keys` (`:238-242`) does `names.sort_unstable()`, so **iteration, `keys()`, `values()` and `items()` stay alphabetical**.
- `__eq__` (`:225-227`) compares two `PyDict`s, and `dict.__eq__` is order-insensitive, so **`as_dict`'s order is not load-bearing for equality** and `__eq__` is unaffected in either direction.

This split is intentional and temporary. Closing it is link 02's job and is explicitly out of scope here.

### 6. Where this lands

This chain lands on the unreleased 0.15 tree. `Cargo.toml` already reads `0.15.0`. This link changes no version literal, adds no pin window, and writes no migration page. Deleting `impl IntoIterator for MetaMap` is a break, and it ships as the 0.15 behaviour. molpy names no Rust `MetaMap`.

The Python split in § 5 is chain order only. It is not a compatibility promise. Link 02 closes it in this same 0.15 tree, and nothing is published between the two links.

### Reuse decision

- `reuse indexmap::IndexMap` — the ordered-map container. Already resolved at 2.14.0 (`Cargo.lock:675-677`); hand-rolling a key-order side table on top of `HashMap` would be a second implementation of a solved problem and would put the ordering invariant in our code instead of a dependency's.
- `reuse MetaValue::to_attr_value` / `from_attr_value` (`meta.rs:154`, `:187`) — the Zarr attribute codec is untouched by this link; the ordering test asserts key order around the existing codec, not a new one.
- `reuse` the `frame_io.rs` test harness — `store_in`, `TempDir`, the `FRAME` const and the `group.attributes().get(...)` assertion shape already in use at `frame_io.rs:1106-1129`. No new fixture module.
- `reuse` the `serialize.rs` `#[cfg(test)] mod tests` harness (`serialize.rs:552-558`) for the wire round-trip ordering test.
- `new — MetaIter<'a>`: nothing in `core::store` wraps a map iterator as a named type, so there is no candidate to reuse or generalize. Its construction and naming follow the closest in-tree pattern, `MetaMap` itself — a tuple newtype over the container, `pub` at `core::store::meta`, re-exported through `core/mod.rs:70` alongside `MetaMap` and `MetaValue`.

## Files to create or modify

- `molrs/Cargo.toml` — add `indexmap = "2"` to the always-on `[dependencies]` block, with the one-line rationale comment the neighbouring entries carry.
- `molrs/src/core/store/meta.rs` — swap the inner container to `IndexMap`; route `remove` to `shift_remove`; add the `MetaIter<'a>` newtype and return it from `iter` and from the borrowed `IntoIterator`; delete the owned `impl IntoIterator for MetaMap`; add the insertion-order rustdoc example; extend `#[cfg(test)] mod tests`.
- `molrs/src/core/mod.rs` — re-export `MetaIter` alongside `MetaMap` and `MetaValue` at line 70.
- `molrs/src/serialize.rs` — **replace both `BTreeMap` meta halves with `IndexMap`**: the serialize side at `:511-513` and the `FrameRepr` deserialize side at `:527`. The two blocks halves (`:509`, `:525`) stay `BTreeMap`. Add a wire round-trip ordering test to `mod tests`.
- `molrs/src/io/zarr/frame_io.rs` — **test module only**; no change to `write_frame_group` or `read_frame_group` themselves. Adds the frame-group attribute-order round-trip test.
- `molrs/src/io/data/cif.rs` — the one owned-iterator consumer. `build_frame` copies `self.meta` through `iter()` into `frame.meta.extend`. No other change.

## Tasks

- [x] Write failing unit tests for insertion-ordered `MetaMap` in `molrs/src/core/store/meta.rs` (`#[cfg(test)] mod tests`): iteration order after out-of-alphabetical inserts, order preserved after `remove` of a middle key, and re-insert keeps the original position
- [x] Write failing wire round-trip ordering test in `molrs/src/serialize.rs` `mod tests`: a frame with out-of-alphabetical meta keys serialized and deserialized yields the same meta key sequence
- [x] Write failing frame-group attribute-order round-trip test in `molrs/src/io/zarr/frame_io.rs` `mod tests`, asserting `group.attributes()` key sequence after `write_frame_group` and the `Frame` meta key sequence after `read_frame_group`
- [x] Add `indexmap = "2"` as an always-on dependency in `molrs/Cargo.toml` (with `features = ["serde"]`, required for the wire `IndexMap`)
- [x] Replace `MetaMap`'s inner `HashMap` with `IndexMap` in `molrs/src/core/store/meta.rs`, routing `remove` to `shift_remove`
- [x] Add the `MetaIter<'a>` newtype in `molrs/src/core/store/meta.rs`, return it from `iter` and the borrowed `IntoIterator`, delete the owned `impl IntoIterator for MetaMap`, and re-export `MetaIter` from `molrs/src/core/mod.rs`
- [x] Replace both `BTreeMap` meta halves with `IndexMap` in `molrs/src/serialize.rs` (serialize side and `FrameRepr` deserialize side), leaving both blocks halves `BTreeMap`
- [x] Add the insertion-order rustdoc example on `MetaMap` per `rustdoc` style, stating that iteration is insertion-ordered and that `remove` preserves the order of the remaining keys
- [x] Point `molrs/src/io/data/cif.rs` `build_frame` at `iter()` + `extend` so deleting the owned iterator still compiles and keeps insertion order
- [ ] Verify `molrs/src/io/data/xyz.rs:1929` and `molrs-wasm/src/core/frame.rs:489` still compile unchanged against the borrowed `IntoIterator` and `keys()` signatures
- [ ] Run full check + test suite (`cargo fmt --check`, `cargo mrs-clippy -- -D warnings`, `cargo mrs-test`, `cargo mrs-doctest`)

## Testing strategy

Unit tests live in `#[cfg(test)]` modules next to the code; there is no `molrs/tests/` tree. Each test targets one function of one module.

**`molrs/src/core/store/meta.rs` — the ordering contract lives here.** Happy path: insert `z`, `a`, `m` and assert `keys()` yields `["z", "a", "m"]`, and that `iter()` and `values()` agree with it. Edge cases: `remove("a")` leaves `["z", "m"]` — the `shift_remove`-versus-`swap_remove` guard, which uses four keys and removes the second, where the two differ; re-inserting an existing key with a new value keeps its original position and updates the value; `clear` then re-insert restarts the order. The existing `MetaValue` JSON tests (`meta.rs:377-444`) are untouched.

**`molrs/src/serialize.rs` — the wire round trip.** One test: build a `Frame` with meta keys in a non-alphabetical, non-hash order, round-trip it through the serde path, and assert the decoded `frame.meta.keys()` sequence equals the original. This is the test that fails if only one of the two meta halves is converted.

**`molrs/src/io/zarr/frame_io.rs` — the Zarr attribute round trip.** One test against `write_frame_group` / `read_frame_group` using the existing `store_in(&dir)` / `TempDir` harness: assert that `Group::open(...).attributes()` yields the frame's meta keys in insertion order, and that `read_frame_group` returns a `Frame` whose `meta.keys()` sequence matches. This also pins the `serde_json` `preserve_order` dependency described in Design § 4 — if that feature is inactive for the gate's feature set, this test is what says so.

No test is added in `molrs/src/io/data/xyz.rs`; see Design § 4 for the recorded reason.

**Regression example.** This repo's regression system is deliberately outside it — `molrs/Cargo.toml:14-18` sets `autoexamples = false` and `CLAUDE.md` § Build & Test Commands states the benchmark and regression systems are being redesigned elsewhere — so there is no `regressions/` directory to write to. The equivalent artifact here is a **rustdoc example on `MetaMap`**: it is public API that compiles and runs, and `cargo mrs-doctest` is in the default gate precisely so such an example cannot rot unnoticed. The example constructs a `MetaMap`, inserts three keys out of alphabetical order, removes one, and `assert_eq!`s the resulting key sequence against a hard-coded literal. No third-party software is involved at any point.

## Out of scope

- **Removing the sorts from `molrs-python`'s `FrameMeta`** (`keys` at `frame.rs:238-242` and the `values`/`items`/`__iter__` that derive from it). That is link 02; the interim split it leaves is documented in Design § 5 rather than papered over.
- **The meta-keyed maps in `molrs/src/io/zarr/sequence.rs`** — `SequenceSchema.meta` (`:755`), the writer meta growth-array map (`:2636`), `PendingFrame.meta` (`:3028`), `resolve_meta` (`:3706`, allocating at `:3718`) and the reader's `metas` (`:4293`). They are **`frame-meta-dict-parity-07-sequence`**, still this 0.15 tree, numbered 07 because `03` is already `untyped-write`. Block, column, cell and mask maps stay `BTreeMap`. Not a later minor.
- **An ordering test in `molrs/src/io/data/xyz.rs`** — behaviour change disclosed in Design § 4, decision recorded there.
- **A contract or rustdoc change for `metaNames()`** in `molrs-wasm` — behaviour change disclosed in Design § 4, decision recorded there.
- **Version literals, pin windows, migration guides, and release notes.** This link does not edit them. Tag and Publish stay on `cgsmiles-03-release` and are not a behaviour task here.
- **Making `Frame`'s block iteration insertion-ordered.** `Frame`'s block map is a separate container with separate consumers (`block_keys`, `blockNames`), and the blocks halves of the wire form stay sorted here. Considered and rejected for this link: it would double the behaviour-change surface without sharing a single line of the diff.
