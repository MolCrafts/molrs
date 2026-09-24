---
spec: frame-meta-dict-parity-01-ordered
created: 2026-09-22
criteria:
  - id: ac-001
    summary: MetaMap's inner container is IndexMap, not HashMap
    type: code
    pass_when: |
      `molrs/src/core/store/meta.rs` declares
      `pub struct MetaMap(IndexMap<String, MetaValue>)` and the file contains
      no `std::collections::HashMap` import or use; `molrs/Cargo.toml` lists
      `indexmap` in the always-on `[dependencies]` block.
    status: pending
  - id: ac-002
    summary: remove() preserves the order of the surviving keys
    type: code
    pass_when: |
      `MetaMap::remove` in `molrs/src/core/store/meta.rs` delegates to
      `shift_remove`, and the string `swap_remove` does not appear in the file.
    status: pending
  - id: ac-003
    summary: No container type appears in MetaMap's public iterator signatures
    type: code
    pass_when: |
      `molrs/src/core/store/meta.rs` contains no `impl IntoIterator for MetaMap`
      (owned) and no `std::collections::hash_map::` path; the borrowed
      `impl<'a> IntoIterator for &'a MetaMap` sets `type IntoIter = MetaIter<'a>`
      and `MetaMap::iter` returns `MetaIter<'_>`; `MetaIter` is re-exported from
      `molrs/src/core/mod.rs`.
    status: pending
  - id: ac-004
    summary: MetaMap iterates in insertion order and survives a middle removal
    type: runtime
    pass_when: |
      The `#[cfg(test)]` tests in `molrs/src/core/store/meta.rs` pass under
      `scripts/test-scope.sh molrs/src/core/store/meta.rs`, including a test that
      inserts four non-alphabetical keys, removes the second, and asserts the
      remaining key sequence equals the hard-coded insertion order minus that
      key, and a test that re-inserting an existing key keeps its original
      position.
    status: pending
  - id: ac-005
    summary: Both serde meta halves are IndexMap; both blocks halves stay BTreeMap
    type: code
    pass_when: |
      In `molrs/src/serialize.rs`, the serialize-side meta binding (currently
      `:511-513`) and the `FrameRepr::meta` field (currently `:527`) are both
      `IndexMap`, while the serialize-side `blocks` binding and
      `FrameRepr::blocks` field are both still `BTreeMap`.
    status: pending
  - id: ac-006
    summary: A serde round trip preserves meta insertion order
    type: runtime
    pass_when: |
      A test in `molrs/src/serialize.rs` `mod tests` builds a Frame whose meta
      keys are inserted in a non-alphabetical order, round-trips it through the
      serde path, and asserts the decoded `frame.meta.keys()` sequence equals
      that hard-coded original sequence; it passes under
      `scripts/test-scope.sh molrs/src/serialize.rs`.
    status: pending
  - id: ac-007
    summary: Zarr frame-group attributes round-trip in meta insertion order
    type: runtime
    pass_when: |
      A test in `molrs/src/io/zarr/frame_io.rs` `mod tests` writes a Frame with
      non-alphabetical meta keys via `write_frame_group`, asserts
      `Group::open(...).attributes()` yields that hard-coded key sequence, and
      asserts `read_frame_group` returns a Frame whose `meta.keys()` sequence
      matches; it passes under
      `scripts/test-scope.sh molrs/src/io/zarr/frame_io.rs`. No production line
      of `write_frame_group` or `read_frame_group` is modified by this link.
      This criterion is also the guard on the `serde_json` `preserve_order`
      assumption recorded in Design section 4: if that feature is inactive for
      the gate's feature set, this is the criterion that fails, and the remedy
      (declaring the feature in `molrs/Cargo.toml`) is inside this link's scope.
    status: pending
  - id: ac-008
    summary: The MetaMap rustdoc example demonstrates the ordering guarantee
    type: runtime
    pass_when: |
      `cargo mrs-doctest` passes, and the rustdoc example on `MetaMap` in
      `molrs/src/core/store/meta.rs` inserts at least three keys out of
      alphabetical order, removes one, and `assert_eq!`s the resulting key
      sequence against a hard-coded literal. The example uses public API only
      and invokes no third-party software. This is this link's regression
      example: `regressions/` does not exist in this repo
      (`molrs/Cargo.toml:14-18`, `CLAUDE.md` section Build and Test Commands),
      so a doctest in the default gate is the equivalent runnable artifact.
    status: pending
  - id: ac-009
    summary: MetaMap rustdoc states the insertion-order and removal guarantees
    type: docs
    pass_when: |
      `MetaMap`'s rustdoc in `molrs/src/core/store/meta.rs` states in prose that
      iteration is insertion-ordered and that `remove` preserves the order of
      the remaining keys.
    status: pending
  - id: ac-010
    summary: Existing meta_ref and metaNames consumers compile unchanged
    type: code
    pass_when: |
      `molrs/src/io/data/xyz.rs:1929` and
      `molrs-wasm/src/core/frame.rs:489-497` are byte-identical to their
      pre-change form, and `cargo mrs-check` plus the molrs-wasm build succeed.
    status: pending
  - id: ac-011
    summary: Full gate is green
    type: runtime
    pass_when: |
      `cargo fmt --check && cargo mrs-clippy -- -D warnings && cargo mrs-test &&
      cargo mrs-doctest` all exit 0.
    status: pending
---

# Acceptance criteria

**ac-001 / ac-002 / ac-003 — the container swap and the API cleanup.** Structural, checkable by reading `meta.rs` and `Cargo.toml`. ac-002 is split out from ac-001 because `shift_remove` versus `swap_remove` is the one detail that silently converts the guarantee into a half-truth: `swap_remove` compiles, passes a naive two-key test, and breaks the order on the first deletion from a map of three or more. ac-003 is what makes the next container change a non-breaking one.

**ac-004 / ac-006 / ac-007 — the three ordering contracts.** One per module that owns a behaviour: `meta.rs` owns the container guarantee, `serialize.rs` owns the wire round trip, `frame_io.rs` owns the Zarr attribute round trip.

There is deliberately **no** criterion for the extended-XYZ comment-line token order (`xyz.rs:1929`) or for `metaNames()` (`molrs-wasm/src/core/frame.rs:489`). Both change their observable output order under this link, both are disclosed in Design section 4, and the decision not to contract them here is recorded there with its reason. ac-010 instead pins that neither call site is *edited*, so the change at those two sites is exactly the container change and nothing else.

**ac-008** is this link's regression example, in the form this repo actually gates.

**ac-011** is the standard closing gate.
