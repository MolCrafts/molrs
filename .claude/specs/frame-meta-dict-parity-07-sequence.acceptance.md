---
spec: frame-meta-dict-parity-07-sequence
created: 2026-09-22
criteria:
  - id: ac-001
    summary: The new test asserts zeta, alpha, mu in that order
    type: code
    pass_when: |
      `molrs/src/io/zarr/sequence.rs` `#[cfg(test)]` contains
      `declared_meta_keys_read_back_in_declaration_order`, which declares meta
      keys `zeta`, `alpha`, `mu` in that order, inserts the frame's meta as
      `mu`, `zeta`, `alpha` with `MetaValue::F64` values `3.0`, `1.0`, `2.0`,
      and `assert_eq!`s the read-back `meta.keys()` sequence to
      `["zeta", "alpha", "mu"]` plus those three paired values. The test does
      not call `strip_schema_attribute` and does not assert block or column
      order.
    status: pending
  - id: ac-002
    summary: SequenceSchema.meta is IndexMap; declare_meta is the only insert
    type: code
    pass_when: |
      `SequenceSchema.meta` in `molrs/src/io/zarr/sequence.rs` is
      `IndexMap<String, MetaSchema>`. The only insert into that field is the
      `None` arm of `declare_meta`. `declare_meta_with_fill`, `from_frames`,
      and `schema_from_store` do not insert into it except by calling
      `declare_meta`. The same-dtype arm stays `Ok(())` without a reinsert.
    status: pending
  - id: ac-003
    summary: Schema PartialEq stays order-independent
    type: code
    pass_when: |
      `SequenceSchema`'s derive list does not include `PartialEq` or `Eq`.
      The hand-written `eq` still returns
      `self.blocks == other.blocks && self.meta == other.meta` and does not
      compare key sequences or `rows_hint`.
    status: pending
  - id: ac-004
    summary: SequenceArrays.meta is IndexMap at the field, create, and open
    type: code
    pass_when: |
      `SequenceArrays.meta` is `IndexMap<String, GrowthArray>`, and both
      `SequenceArrays::create` and `SequenceArrays::open` construct that map
      with `IndexMap::new()`. Neither site uses `BTreeMap` for meta.
    status: pending
  - id: ac-005
    summary: PendingFrame.meta and resolve_meta use MetaMap
    type: code
    pass_when: |
      `PendingFrame.meta` has type `MetaMap`. `resolve_meta` is still named
      `resolve_meta` and returns `Result<MetaMap, MolRsError>`. The file
      `molrs/src/io/zarr/sequence.rs` contains no `IndexMap<String, MetaValue>`.
    status: pending
  - id: ac-006
    summary: ReadState.metas is an IndexMap of zarr arrays
    type: code
    pass_when: |
      `ReadState.metas` is
      `IndexMap<String, Array<dyn ReadableListableStorageTraits>>`, not
      `MetaMap` and not `GrowthArray`.
    status: pending
  - id: ac-007
    summary: Non-meta BTreeMaps in sequence.rs stay BTreeMap
    type: code
    pass_when: |
      In `molrs/src/io/zarr/sequence.rs`, `SequenceSchema.blocks`,
      `SequenceSchema.rows_hint`, `BlockSchema.columns`,
      `SequenceArrays.blocks`, `PendingFrame.blocks`, `ReadState.columns`,
      `ReadState.masks`, `FrameSequence.blocks`, `BoxReader.cells`, and the
      test helpers `file_map` and `chunk_files` are still `BTreeMap`. The
      only maps that left `BTreeMap` are `SequenceSchema.meta`,
      `SequenceArrays.meta`, `PendingFrame.meta`, the map `resolve_meta`
      allocates, and `ReadState.metas`.
    status: pending
  - id: ac-008
    summary: indexmap serde is the only Cargo.toml edit
    type: code
    pass_when: |
      `molrs/Cargo.toml`'s `indexmap` dependency enables feature `serde` and
      no other feature, at version `"2"`, with default features left on.
      `serde_json` is still `serde_json = "1"` with no features. The package
      version is unchanged. `Cargo.lock`'s `indexmap` package lists `serde`
      among its dependencies.
    status: pending
  - id: ac-009
    summary: Declared meta keys read back as zeta, alpha, mu
    type: runtime
    pass_when: |
      `declared_meta_keys_read_back_in_declaration_order` passes under
      `scripts/test-scope.sh molrs/src/io/zarr/sequence.rs`. Read-back keys
      are the hard-coded sequence `zeta`, `alpha`, `mu` with values
      `1.0`, `2.0`, `3.0`. This is the regression example: `regressions/`
      does not exist in this repo (`molrs/Cargo.toml` `autotests = false`,
      `autoexamples = false`; `CLAUDE.md` Testing Rules), and this link must
      not create one. No third-party program is invoked.
    status: pending
  - id: ac-010
    summary: meta_keys rustdoc states declaration order
    type: docs
    pass_when: |
      The rustdoc on `SequenceSchema::meta_keys` in
      `molrs/src/io/zarr/sequence.rs` states that keys are yielded in
      declaration order, that `from_frame` / `from_frames` follow
      `frame.meta` iteration order, and that a pinned read inserts into the
      returned frame in that same order.
    status: pending
  - id: ac-011
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `cargo fmt --check && cargo mrs-clippy -- -D warnings && cargo clippy --manifest-path molrs-cxxapi/Cargo.toml --all-targets -- -D warnings && cargo mrs-test && cargo mrs-doctest`
      all exit 0.
    status: pending
out_of_scope:
  - "Link 01's MetaMap container swap, MetaIter, and the Frame serde wire form"
  - "Binder sorts, the Python dtype rule, dict views, frozen MetaDocument, release notes"
  - "Block, column, mask, cell, rows_hint, and file_map ordering"
  - "A format-migration shim, dual map, sorted-order flag, version bump, pin, or migration doc"
  - "Declaration order for a pin-stripped schema_from_store listing"
  - "Edits under molrs-python or molrs-cxxapi"
  - "preserve_order on molrs's direct serde_json dependency"
  - "A regressions/ script or a third-party oracle"
---

# Acceptance — frame-meta-dict-parity-07-sequence

Done means a pinned sequence round-trips meta keys in declaration order, the five meta-key maps are the containers named above, every other `BTreeMap` in `sequence.rs` is untouched, and schema equality still ignores order. The round trip is one unit test. Three of the five maps are not observable through that test, so they are pinned by reading the types.

## AC-001 — The new test asserts zeta, alpha, mu in that order

The test is the RED witness. Declaration order is `zeta`, `alpha`, `mu`. The frame inserts `mu`, `zeta`, `alpha` so a pass cannot be explained by echoing the frame. Alphabetical order, which is what `BTreeMap` returns today, is `alpha`, `mu`, `zeta`. Values stay tied to keys so a swap of values under the right keys fails. The pin stays in place: `strip_schema_attribute` is the uncontracted path.

## AC-002 — SequenceSchema.meta is IndexMap; declare_meta is the only insert

`IndexMap<String, MetaSchema>`, not `MetaMap`, because the value is `MetaSchema`. One insert door, so `from_frames` and `schema_from_store` cannot grow a second order.

## AC-003 — Schema PartialEq stays order-independent

`IndexMap`'s `PartialEq` already ignores order. Putting `PartialEq` on the derive would also start comparing `rows_hint`, which the hand-written `eq` exists to exclude. Both mistakes are refused here.

## AC-004 — SequenceArrays.meta is IndexMap at the field, create, and open

Both constructors allocate the map. Changing only the field type would not compile; changing only one constructor would leave a reopen sorted. The value type stays `GrowthArray`.

## AC-005 — PendingFrame.meta and resolve_meta use MetaMap

`resolve_meta` keeps its name and returns `MetaMap`. A parallel `IndexMap<String, MetaValue>` would be a second meta map beside the one link 01 just made ordered. The file must contain no such type.

## AC-006 — ReadState.metas is an IndexMap of zarr arrays

The reader cache holds `Array<dyn ReadableListableStorageTraits>`, not `MetaValue` and not `GrowthArray`. Order matches `schema.meta` because `frame()` inserts into this map while walking the schema; the type is what keeps a later walk from re-sorting.

## AC-007 — Non-meta BTreeMaps in sequence.rs stay BTreeMap

Block order was left sorted by link 01 on purpose. This criterion is the grep that stops a whole-file conversion. The five meta-key maps are the only ones allowed to leave `BTreeMap`.

## AC-008 — indexmap serde is the only Cargo.toml edit

Verified while drafting: `indexmap` 2.14.0 in `Cargo.lock` does not depend on `serde`, and `SequenceSchema`'s derive needs `IndexMap: Serialize + Deserialize`, which is the `serde` feature. `serde_json`'s `preserve_order` does not turn that feature on. `zarrs` 0.23.13 already enables `preserve_order`, and this file compiles only with `zarr`, so molrs's own `serde_json` line is not given features. The lock refresh is cargo's, and it must list `serde` under `indexmap`.

## AC-009 — Declared meta keys read back as zeta, alpha, mu

The test goes green only when the pin survives `to_value` / `from_value` and `frame()` inserts in schema order. This criterion is also the regression example. `regressions/` is a cited absence, not an oversight: `molrs/Cargo.toml` disables integration tests and examples, and `CLAUDE.md` keeps unit tests next to the code.

## AC-010 — meta_keys rustdoc states declaration order

`meta_keys` is the public iterator of the pin. The rustdoc is prose, not a store-opening doctest. It is the sentence a caller reads instead of assuming alphabetical order.

## AC-011 — Full check and test suite pass

The gate from `CLAUDE.md`: format, `cargo mrs-clippy -D warnings`, the cxxapi clippy (it constructs `SequenceSchema` and must still build), `cargo mrs-test`, and `cargo mrs-doctest`.
