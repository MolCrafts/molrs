---
title: Sequence meta keys follow declaration order
slug: frame-meta-dict-parity-07-sequence
status: in-progress
created: 2026-09-22
chain: frame-meta-dict-parity (01-ordered → 02-binder-order → 03-untyped-write → 04-dict-views → 05-document → 07-sequence)
depends_on: frame-meta-dict-parity-01-ordered
---

# Sequence meta keys follow declaration order

## Summary

Per-step metadata on a frame sequence is stored in `BTreeMap`s, so declaration order is thrown away and every reader rebuilds keys alphabetically. This link changes only the meta-key maps in `molrs/src/io/zarr/sequence.rs` so a sequence keeps declaration order — the order `declare_meta` first inserted each key, which `from_frame` / `from_frames` take from `frame.meta` — from the pinned schema through the writer’s buffers to the frame a reader returns. Schema equality stays order-independent. The work lands in unreleased 0.15 and assumes link 01 has already made `MetaMap` an insertion-ordered `IndexMap`.

## Domain basis

None. This link declares no physics: no equation, no constant, no unit, and no literature reference is involved. `$META.science.required` is true for molrs; the section is stated and empty by that rule, not skipped. Step times stay femtoseconds where the writer already documents them; this link does not touch time.

## Design

**Constitution.** `.claude/notes/law.md` does not exist in this repo. The governing rules are `CLAUDE.md` § *Design preferences* (no silent debt, OOP by default, primitive single-responsibility APIs, the shape check) and § *Testing Rules*, `.claude/notes/architecture-rules.md` (single `molcrafts-molrs` crate, `io` may name `core`, no dual public names), and `.claude/notes/testing.md` (unit tests live in `#[cfg(test)]` next to the code; there is no `molrs/tests/` tree and no `regressions/` tree).

**What link 01 has already done, and what this link must not redo.** `MetaMap` iterates in insertion order, `remove` is `shift_remove`, and `extend` exists. `indexmap` is already a direct always-on dependency in `molrs/Cargo.toml`. This link consumes both. It does not re-specify the container swap, does not add a second ordered-map type, and does not touch `molrs/src/core/store/meta.rs`.

**Release.** The crate version is already the unreleased `0.15.0`. No semver bump, no pin, no migration document, no release notes, no compatibility shim, and no dual map. Stores already written by this unreleased tree pinned meta keys in sorted order; they read back in that stored order. Nothing rewrites them.

### 1. The five meta-key maps, and only those

Every map below is private. No public signature changes: `meta_keys` stays `impl Iterator<Item = (&str, &str)>`, and `SequenceSchema`’s fields stay private. Callers outside this file (`molrs-python`, `molrs-cxxapi`) compile unchanged.

| Site | Today | After this link |
|---|---|---|
| `SequenceSchema.meta` (currently `:755`) | `BTreeMap<String, MetaSchema>` | `IndexMap<String, MetaSchema>` |
| `SequenceArrays.meta` (currently `:2636`), built in `create` (`:2668`) and `open` (`:2813`) | `BTreeMap<String, GrowthArray>` | `IndexMap<String, GrowthArray>` |
| `PendingFrame.meta` (currently `:3028`) | `BTreeMap<String, MetaValue>` | `MetaMap` |
| `resolve_meta` (currently `:3706`, allocates `:3718`, called `:3581`) | returns `BTreeMap<String, MetaValue>` | keeps the name `resolve_meta`, returns `MetaMap` |
| `ReadState.metas` (currently `:4293`) | `BTreeMap<String, Array<dyn ReadableListableStorageTraits>>` | `IndexMap` of that same array type |

`PendingFrame.meta` and `resolve_meta` use `MetaMap`, not a parallel `IndexMap<String, MetaValue>`. `ReadState.metas` stores open zarr arrays, so it is an `IndexMap` of the existing array type, not a `MetaMap`. `SequenceArrays.meta` stays `GrowthArray`; the reader’s cache stays `Array`. Do not unify those two value types.

`declare_meta` (currently `:1027`, insert at `:1041`) stays the only door that inserts into `SequenceSchema.meta`. `declare_meta_with_fill` (`:1064`), `from_frames` (`:858-865`), and `schema_from_store` (`:1248`) keep calling it; they do not gain their own insert. The match arms stay: an unknown dtype is an error, the same key with another dtype is an error, the same key with the same dtype is `Ok(())` and does **not** reinsert, and a new key is inserted at the end. A second `declare_meta` of an existing key therefore keeps its original position. Do not add a flag that restores sorted order.

`SequenceSchema` keeps its hand-written `PartialEq` (currently `:761-766`): `self.blocks == other.blocks && self.meta == other.meta`, with `rows_hint` excluded. Do not add `PartialEq` to the derive — that would start treating row hints as identity. `IndexMap` and `MetaMap` already compare as maps, order-independently, so two schemas that declare the same keys in different orders stay equal. No order-sensitive comparison is added.

The loops that already walk these maps stay loops. Do not sort them and do not rewrite them:

- `from_frames` walks `frame.meta` (insertion order, once link 01 has landed) and declares each new key through `declare_meta`. First-seen order across the slice is the derived schema’s meta order. `step` and `time` stay skipped.
- `schema_from_store` still declares in `children()` order. See § 3.
- `SequenceArrays::create` and `open`, `resolve_meta`, the commit loop (currently `:3977`), and `FrameSequence`’s frame read (currently `:4632-4643`) keep iterating `schema.meta` and looking up the other maps by key.

`resolve_meta` builds the `MetaMap` by inserting in `schema.meta` order, not in the frame’s own meta order. A frame whose keys were inserted differently does not reorder the sequence. The read loop inserts into `frame.meta` in that same schema order, which is what a caller sees on `keys()`.

Add `use indexmap::IndexMap;` at the top of `sequence.rs`, and widen the existing meta import to `use molrs::store::meta::{MetaMap, MetaValue};`. `use std::collections::{BTreeMap, VecDeque}` stays: the maps in § 2 still use it.

### 2. Maps that stay `BTreeMap`

Link 01 left block-name order sorted on purpose. This link does not reopen that. These stay `BTreeMap`, including every local and test helper that is not a meta-key map:

- `BlockSchema.columns`, `SequenceSchema.blocks`, `SequenceSchema.rows_hint`, and the `shapes` map inside `from_frames`
- column and mask `GrowthArray` maps, `SequenceArrays.blocks`, `PendingFrame.blocks`, `landed_blocks` (the writer field and `Attached`)
- `BoxReader.cells`, `ReadState.columns`, `ReadState.masks`, `FrameSequence.blocks`, and the column-selection `wanted` map inside the reader
- test helpers `file_map` and `chunk_files`

A drive-by conversion of any of those is a failure of this link, not a bonus.

### 3. Where order is visible, and where it is not

The user-visible order is the schema’s. `meta_keys` yields it. A reader that opens a store **with** the `sequence_schema` pin yields it on `frame.meta`, because `frame()` walks `schema.meta` and `MetaMap::insert` keeps that sequence.

The pin is the order carrier across a process boundary. `FrameSequenceWriter::create_with` stores it with `serde_json::to_value(&schema)` (currently `:3195-3198`); `pinned_schema` reads it back with `serde_json::from_value` (currently `:1133`). Both directions preserve `IndexMap` entry order only because of the two facts verified below. The round-trip test is the guard on top of that verification.

**Verified, so this link does not guess.**

- `Cargo.lock` resolves `indexmap` 2.14.0 with dependencies `equivalent` and `hashbrown` only. The `serde` feature is off. Registry `indexmap` 2.14.0 defines `serde = ["dep:serde_core", "dep:serde"]`. `SequenceSchema` derives `Serialize` and `Deserialize`, so naming `IndexMap` on `meta` does not compile until that feature is on. A plain `use indexmap::IndexMap` compiles once link 01 has added the dep; the derive does not.
- `serde_json` 1.0.151 defines `preserve_order = ["indexmap", "std"]`. That enables the `indexmap` dependency. It does **not** enable `indexmap`’s `serde` feature, which is why the derive is still unsatisfied.
- `zarrs` 0.23.13 depends on `serde_json` with `preserve_order` (and `float_roundtrip`). `sequence.rs` is compiled only under `feature = "zarr"` (`molrs/src/io/mod.rs`), which is the feature that turns `zarrs` on, and `default` / `cargo mrs-test` turn `zarr` on through `filesystem`. Every build that typechecks this file therefore builds `serde_json::Map` as an order-preserving map. `to_value` / `from_value` of the pin keep entry order. Do **not** add `preserve_order` to molrs’s own `serde_json = "1"` line.

The one manifest edit is on the `indexmap` line link 01 added: `features = ["serde"]`, default features left on, version still `"2"`. No other line in `molrs/Cargo.toml` changes, and the workspace version stays `0.15.0`. Do not hand-edit `Cargo.lock`; cargo’s refresh, which adds `serde` under the `indexmap` package, is committed with that line.

`SequenceArrays.meta`, `PendingFrame.meta`, and `ReadState.metas` are not what the reader iterates to build `frame.meta`. The write loop and the read loop both walk `schema.meta` and look up the other maps by key. Their container change is what stops a later iteration of those maps from silently re-sorting. The round-trip test does not observe them; the type criteria do.

`schema_from_store` is the pin-stripped fallback. It still declares meta keys in `children()` order, and this link does not reorder directory listings or invent a sidecar to recover declaration order. A store whose `sequence_schema` attribute was removed is not contracted to declaration order. The new test must not call `strip_schema_attribute`.

**Disclosed, not contracted, not edited.** `molrs-python/src/io/mrec.rs` `meta_keys` (currently `:449`) collects `SequenceSchema::meta_keys()` into a `Vec` with no sort of its own, and `python/molrs/io/mrec/__init__.py` returns that list. Its order follows this link without a Python change. No binder file is modified. Frame-meta listing on the Python side is link 02’s surface and is not re-specified here.

### 4. `meta_keys` says so

`SequenceSchema::meta_keys` (currently `:1087`) is the public iterator of the pin. Its rustdoc gains one or two sentences: keys are yielded in declaration order, the order `declare_meta` first inserted each key; `from_frame` / `from_frames` declare in `frame.meta`’s iteration order; reading a pinned sequence inserts into the returned frame in this same order. No runnable doctest — a doctest would have to open a store, and the executable contract is the unit test below. Do not document block or column order.

### Reuse decision

The container choices below are the librarian’s, closed by the operator. Each is resolved here and nowhere else.

- `reuse indexmap::IndexMap` — `SequenceSchema.meta` (`IndexMap<String, MetaSchema>`, not `MetaMap`), `SequenceArrays.meta` (`IndexMap<String, GrowthArray>`), and `ReadState.metas` (`IndexMap` of the existing `Array<dyn ReadableListableStorageTraits>`). Link 01 already added the dependency; this link only turns on `serde`.
- `reuse MetaMap` — `PendingFrame.meta` and the value `resolve_meta` returns. Same key and value as today’s map. Do not invent `IndexMap<String, MetaValue>`.
- `reuse declare_meta` — the only insert into `SequenceSchema.meta`. `declare_meta_with_fill` stays a fill written onto the entry `declare_meta` just inserted.
- `reuse` the existing `#[cfg(test)]` helpers in `sequence.rs` — `store_in`, `TempDir`, `atoms_frame`, `FrameSequenceWriter::create` / `append` / `close`, `FrameSequence::open`, `frame`. No new fixture module.
- `new — none.` No new type, no new free function, no second way to declare a meta key.

## Files to create or modify

- `molrs/src/io/zarr/sequence.rs` — the five meta-key maps, the `IndexMap` / `MetaMap` imports, the `meta_keys` rustdoc, and one test in the existing `#[cfg(test)]` module. No other `BTreeMap` in the file changes.
- `molrs/Cargo.toml` — the only manifest edit: add `features = ["serde"]` to the `indexmap` dependency link 01 added. Nothing else in the file, including the version, moves.
- `Cargo.lock` — cargo’s refresh of that feature only (the resolved `indexmap` package gains `serde`). Not hand-edited.

No other file. `SequenceSchema.meta`, `resolve_meta`, `PendingFrame`, `SequenceArrays`, and `ReadState` are private to `sequence.rs`. Binders call `declare_meta` / `from_frame` and never name these map types.

## Tasks

- [x] Write a failing unit test `declared_meta_keys_read_back_in_declaration_order` in the existing `#[cfg(test)]` module of `molrs/src/io/zarr/sequence.rs`, next to the meta round-trip tests: declare `zeta`, `alpha`, `mu` in that order, write one frame whose meta was inserted in a different order, and assert the read-back key sequence is the declaration order
- [x] Enable `features = ["serde"]` on the `indexmap` dependency in `molrs/Cargo.toml`, leaving cargo to refresh `Cargo.lock`
- [x] Change `SequenceSchema.meta` to `IndexMap<String, MetaSchema>` in `molrs/src/io/zarr/sequence.rs`, add `use indexmap::IndexMap`, keep `declare_meta` as the only insert, and leave `SequenceSchema::eq` as it is
- [x] Change `SequenceArrays.meta` to `IndexMap<String, GrowthArray>` at the field and at both constructions (`create` and `open`) in `molrs/src/io/zarr/sequence.rs`
- [x] Change `PendingFrame.meta` and `resolve_meta` to `MetaMap` in `molrs/src/io/zarr/sequence.rs`, importing `MetaMap` beside `MetaValue`, without introducing `IndexMap<String, MetaValue>`
- [x] Change `ReadState.metas` to `IndexMap<String, Array<dyn ReadableListableStorageTraits>>` in `molrs/src/io/zarr/sequence.rs`
- [x] State declaration order on the `SequenceSchema::meta_keys` rustdoc in `molrs/src/io/zarr/sequence.rs`
- [ ] Run full check + test suite (`cargo fmt --check && cargo mrs-clippy -- -D warnings && cargo clippy --manifest-path molrs-cxxapi/Cargo.toml --all-targets -- -D warnings && cargo mrs-test && cargo mrs-doctest`)

## Testing strategy

Unit tests live in the `#[cfg(test)]` module of `molrs/src/io/zarr/sequence.rs`. There is no `molrs/tests/` tree. The new test follows the module’s existing meta tests: one write through `FrameSequenceWriter`, one read through `FrameSequence::frame`, hand-written inputs, no external program. Inner loop: `scripts/test-scope.sh molrs/src/io/zarr/sequence.rs`.

**The one new test, `declared_meta_keys_read_back_in_declaration_order`.** It is written first and fails on current `BTreeMap` order. Using the existing `store_in` / `atoms_frame` helpers:

- `SequenceSchema::new()`, `declare_column` of the stock `atoms` / `x` `f64` column, then `declare_meta("zeta", "f64")`, `declare_meta("alpha", "f64")`, `declare_meta("mu", "f64")`.
- One frame carrying that column, with meta inserted as `mu = F64(3.0)`, `zeta = F64(1.0)`, `alpha = F64(2.0)` — neither alphabetical nor declaration order.
- `FrameSequenceWriter::create`, `append`, `close`, then `FrameSequence::open` and `frame(0)`.
- `assert_eq!` the collected `meta.keys()` sequence to `["zeta", "alpha", "mu"]`, and `assert_eq!` `get("zeta")` / `get("alpha")` / `get("mu")` to `Some(&MetaValue::F64(1.0))`, `Some(&MetaValue::F64(2.0))`, `Some(&MetaValue::F64(3.0))`.

Alphabetical read-back (`alpha`, `mu`, `zeta`) is the current failure. The test does not call `strip_schema_attribute`, does not assert `block_names` or column order, and does not import a third-party oracle. No second test is added: re-declare-keeps-position and order-independent `PartialEq` are pinned by leaving those arms and that `eq` body unchanged, and the other three maps are not visible through this read.

**Regression example.** Cited exception, same as link 01 and as `CLAUDE.md` / `.claude/notes/testing.md`: this repo sets `autotests = false` and `autoexamples = false` (`molrs/Cargo.toml`) and has no `regressions/` tree. Do not create one. The unit test above is the runnable artifact, with the key sequence and the three values as literals.

Existing meta tests (`every_meta_variant_round_trips_bit_exact_with_its_dtype_tag`, the fill tests, `a_meta_value_at_another_width_is_read_at_the_declared_one`) use `get` on a single key and stay as they are.

## Out of scope

- Link 01’s `MetaMap` container swap, `MetaIter`, and the serde wire form of `Frame`. Consumed, not redone.
- Binder sorts (link 02), the Python dtype rule (link 03), dict views (link 04), frozen `MetaDocument` (link 05), and release notes.
- Block, column, mask, cell, `rows_hint`, and `file_map` ordering. Those `BTreeMap`s stay `BTreeMap`. No test asserts that block or column order changed.
- A format-migration shim, a dual map, a sorted-order flag, a version bump, a pin, or a migration document.
- Making `schema_from_store` recover declaration order from a pin-stripped listing. Child order is unchanged and uncontracted.
- Edits under `molrs-python/`, `molrs-cxxapi/`, or any other crate. The Python `meta_keys` order change is the disclosure in § 3, not a task.
- `preserve_order` on molrs’s direct `serde_json` dependency. Verified already active via `zarrs` for every build that compiles this file.
- A `regressions/` script or a third-party oracle.
