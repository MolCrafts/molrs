---
title: A plain write to frame.meta takes the value's own dtype
slug: frame-meta-dict-parity-03-untyped-write
status: in-progress
created: 2026-09-22
chain: frame-meta-dict-parity (01-ordered → 02-binder-order → 03-untyped-write → 04-dict-views → 05-document → 07-sequence)
revised: 2026-09-22
depends_on:
---

# A plain write to frame.meta takes the value's own dtype

## Summary

`frame.meta["k"] = v` in the Python binder currently looks up whatever dtype key `k` already holds and coerces `v` into it, raising `TypeError` when the value does not fit. This link deletes that rule. A plain write infers the value's own dtype, exactly as every other binder already does, so `frame.meta` behaves like the `MutableMapping` it advertises. `MetaValue("f32", 300.0)` remains the one way to pin a non-default dtype, and the two paths on which a dtype tag is actually durable — a declared sequence schema and the serde frame document — are stated in every place the old rule was stated, so a user who needs `f32` to survive a write knows where to say so.

## Domain basis

None. This link changes a metadata dtype rule at a language boundary; no equation, unit or physical quantity is involved, so `$META.science.required` raises no Domain-basis obligation here. Stated rather than silently omitted.

## Design

### What is deleted

`PyFrameMeta::typed_for` (`molrs-python/src/core/store/frame.rs:134-147`) is removed. Its four call sites — `__setitem__` (`:202`), `absorb` (`:178`, `:185`) and `setdefault` (`:316`) — call `infer_meta_value` directly. This is behaviour-preserving for tagged writes: `typed_for`'s first arm (`:136-138`) already delegated a `PyMetaValue` to `infer_meta_value`, and `infer_meta_value` itself unwraps a `PyMetaValue` first (`:869-872`). Only the untagged arm (`:139-146`, the `tag_of` lookup plus the `"assign a MetaValue to change it"` `TypeError`) disappears.

No new symbol is introduced, so the shape check does not apply; this is a deletion that restores an existing method to the one behaviour its siblings already have.

### Why the rule was never an invariant

Two independent facts, both verified in tree:

1. **Every other binder already infers.** `molrs-wasm`'s `setMeta` inserts `MetaValue::String` and `setMetaScalar` inserts `MetaValue::F64` without consulting the key (`molrs-wasm/src/core/frame.rs:543-555,558-566`); `molrs-capi`'s `meta_from_c` builds from the caller's own dtype tag (`molrs-capi/src/frame.rs:732-739`); `molrs-cxxapi`'s `frame_set_meta_entry` inserts the decoded entry as-is (`molrs-cxxapi/src/lib.rs:1128-1135`). The typed-slot rule existed in exactly one binder, so deleting it *restores* binding-surface symmetry (`.claude/notes/notes.md`, 2026-08-10).
2. **It is breachable one line away, inside the same class.** `mapping_to_meta_map` (`molrs-python/src/core/store/frame.rs:78-97`) re-infers every value unless the source is a `PyFrameMeta`, so `frame.meta = dict(frame.meta)` already widens an `f32` key to `f64` today. A rule a caller breaks by round-tripping through `dict` is a speed bump, not an invariant.

A cross-repo consumer sweep found nothing that depends on key pinning: molpy and molvis construct `MetaValue(...)` explicitly (`molvis/python/src/molvis/wire.py:536-538`, `molpy/tests/…`), molpy's LAMMPS trajectory reader writes plain values to fresh keys (`molpy/src/molpy/io/trajectory/lammps.py:36-71`), and molrec goes through `declare_meta` (`molrec/tests/molrs_adapter.py:155-157`).

### Where a dtype tag is durable — exactly two paths

The replacement prose must bound durability to both of these, and to no more:

1. **A declared sequence schema.** `SequenceSchema::declare_meta(key, dtype)` pins the dtype; `FrameSequenceWriter::resolve_meta` (`molrs/src/io/zarr/sequence.rs:3706-3756`) re-reads an incoming value at the declared width (`:3726-3729`) and refuses one that width cannot hold, naming both dtypes (`:3730-3736`). This is what makes a plain Python `float` land in a declared `f32` column.
2. **The serde frame document.** `impl Serialize for MetaValue` (`molrs/src/serialize.rs:37-41`) emits the typed `{dtype, value}` envelope via `to_json_value`, and `Deserialize` (`:43-48`) re-reads it through `from_json_value`. `impl Serialize for Frame` (`:504-519`) puts that map on every frame document at `version: 2`, and `serialize.rs:587-608` already asserts the tag survives a round trip. It is reachable from Python through `molrs.io.write_frame_bytes` / `read_frame_bytes` (`molrs-python/src/io/mod.rs:2536-2558`, stubs at `molrs-python/python/molrs/_lib.pyi:1402-1405`), and `stream` is unconditional in `molrs-python/Cargo.toml:45`, so the path is always compiled in.

Everywhere else the tag is **already** not durable, before this change and after it: a per-frame Zarr group writes `to_attr_value()` — the plain, untagged payload — into the group's attributes (`molrs/src/io/zarr/frame_io.rs:536`) and reads it back through `MetaValue::from_attr_value`, which infers (`:640`); and `MolRec::meta` is a plain `JsonMap<String, JsonValue>` (`molrs/src/core/store/record.rs:100`). Naming these as already-untagged is what makes the two-path bound exact rather than merely a list of the paths that happen to be in scope.

### What a consumer sees change

Post-deletion, on a key that previously held `f32`, `frame.meta["scale"] = 0.5` followed by `write_frame_bytes(frame)` emits `{"dtype":"f64","value":0.5}` where it used to emit `{"dtype":"f32","value":0.5}`, and the receiver's `read_frame_bytes` yields `MetaValue::F64`. The fix on the writer's side is `frame.meta["scale"] = MetaValue("f32", 0.5)`; on a declared sequence the schema already re-narrows it and nothing changes. The one checked wire consumer, `molvis/python/src/molvis/wire.py:536-538`, wraps every incoming meta value as `MetaValue("f64", float(v))` explicitly and is therefore unaffected.

**The coercion boundary, stated correctly.** At serialization into a declared sequence schema the value is coerced to the declared width, and errors **only** when the declared dtype cannot represent it at all. `resolve_meta` re-reads through `MetaValue::from_json_value`, whose `f32` arm (`molrs/src/core/store/meta.rs:239-246`) is `let narrowed = raw as f32`, erroring only when a finite input goes non-finite — so a declared `f32` carrying `310.0` is **silently narrowed**, not refused. An error appears only where the declared width genuinely cannot hold the value: declared `i64` against a frame float fails `as_i64()` (`meta.rs:232`). The sequence test must use the declared-`i64` case — one that actually raises — and must not promise a write-time refusal for float widths.

### Coverage: correcting this spec's own premise

An earlier draft of this spec claimed `resolve_meta` has no test anywhere. **That is false and is corrected here.** `molrs/src/io/zarr/sequence.rs:7194-7223`, `a_meta_value_at_another_width_is_read_at_the_declared_one`, declares `scale` as `f32`, appends a frame carrying `MetaValue::F64(0.5)` (`:7206`) and asserts `back.meta.get("scale") == Some(&MetaValue::F32(0.5))` (`:7222`). It runs in the default gate (`#[cfg(all(test, feature = "filesystem"))]`; the `mrs-*` aliases pass `filesystem`).

What is genuinely uncovered is **refusal by width**: the existing failure assertion (`:7208-7214`) trips on a *shape* mismatch — a two-element list under a declared `f64x3` — not on a value the declared width cannot hold. The test's own doc comment (`:7191-7193`) promises "one that cannot be is refused naming both", so the gap is inside a promise already made. The fix is therefore an extension of that test, not a new one: declare a third key `count` as `i64`, carry it on both existing frames, and append one further frame whose `count` is `MetaValue::F64(0.5)`. `"i64" => payload.as_i64()` returns `None` for a JSON float, so `resolve_meta` produces `meta key "count" is declared i64 but this frame carries f64: …`. The third append is its own call because `resolve_meta` walks `self.schema.meta` in `BTreeMap` order, where `"com"` precedes `"count"` and would mask it.

That is the **only** hunk this link places under `molrs/src/`, and it is inside a `#[cfg(test)]` module.

### Replacement prose (the exact claim every site must carry)

> `dtype(k)` reports the tag of the value stored right now; any plain write re-infers it. `MetaValue` fixes the dtype of that write only — it does not pin the key. A tag survives a round trip only through a declared sequence schema or the serde frame document; outside those two it is re-inferred on read.

Both `frame.rs:100-115` and `_lib.pyi:333-343` carry the non-sticky qualifier; neither may describe `MetaValue` as "how a key is given a dtype" without it.

### Reuse decision

No `librarian_report` was supplied (blueprint refresh deferred). Resolved from the tree directly:

- `reuse infer_meta_value` (`frame.rs:869`) — the four former `typed_for` call sites call it. No inference logic is written.
- `reuse meta_value_from_dtype` (`frame.rs:823`) — unchanged, two live callers remain (`PyMetaValue::new` at `:44`, the fixed-length-vector branch of `infer_meta_value` at `:902`), so the deletion leaves no dead code.
- `reuse tag_of` / `dtype()` / `typed()` — unchanged. The reader surface stays coherent; what is gone is its durability, not its meaning. `typed()` is load-bearing for pickle and copy.
- `reuse a_meta_value_at_another_width_is_read_at_the_declared_one` (`molrs/src/io/zarr/sequence.rs:7195`) — extended in place rather than paralleled by a second test.
- `MetaMap::validate_against(schema)` in `core` — **new — rejected**: `core` depends on no other molrs module and owns Frame/Block/schema, not the write-time policy of a sequence writer. No such symbol is added.

### Why this is one link and not two

Files touch two crates (`molrs-python` and `molrs`), which normally argues for a split. The `molrs` side is a single test-function extension that adds no symbol and changes no behaviour, and it exists only because this link's own Design premise about that test was wrong. Splitting a three-assertion test extension into its own spec would cost more than it proves. Recorded here so the exception is visible rather than silent.

## Files to create or modify

- `molrs-python/src/core/store/frame.rs` — delete `typed_for`; route `__setitem__` / `absorb` / `setdefault` through `infer_meta_value`; rewrite the `PyFrameMeta` doc (`:100-115`) and the `mapping_to_meta_map` doc (`:74-77`) to state the two durable paths.
- `molrs-python/python/molrs/_lib.pyi` — rewrite the `FrameMeta` docstring (`:333-343`).
- `molrs-python/tests/test_frame.py` — replace the two typed-slot tests (`:193-199`, `:213-217`); narrow the claim of `test_reassigning_a_read_value_is_an_identity` (`:201-211`).
- `molrs/src/io/zarr/sequence.rs` — **test-only**; extend `a_meta_value_at_another_width_is_read_at_the_declared_one` (`:7194-7223`) with the declared-`i64` refusal.
- `.claude/notes/notes.md` — a new entry recording the untagged frame-group / `MolRec` meta path.

## Tasks

- [x] Rewrite the three dtype tests in `molrs-python/tests/test_frame.py::TestFrameMeta` to assert re-inference: rename `test_existing_key_keeps_its_dtype` to `test_a_plain_write_takes_the_values_own_dtype`; replace `test_value_that_does_not_fit_the_slot_is_refused` with `test_a_plain_write_replaces_the_slots_dtype`; narrow `test_reassigning_a_read_value_is_an_identity` to the inferred-default dtypes (`i64`, `f64x6`) and add an `f32` case asserting the tag does **not** survive
- [x] Extend `a_meta_value_at_another_width_is_read_at_the_declared_one` in `molrs/src/io/zarr/sequence.rs` with a declared-`i64` key carried on the existing frames plus one further append whose value is `MetaValue::F64(0.5)`, asserting the error names both `"count"` and `i64`
- [x] Delete `PyFrameMeta::typed_for` in `molrs-python/src/core/store/frame.rs` and call `infer_meta_value` at `__setitem__`, both `absorb` arms and `setdefault`
- [x] Rewrite the `PyFrameMeta` and `mapping_to_meta_map` rustdoc in `molrs-python/src/core/store/frame.rs` per `doc.style`: a plain write infers, `MetaValue` pins one write, a tag is durable on exactly the two named paths
- [x] Update the `FrameMeta` docstring in `molrs-python/python/molrs/_lib.pyi` to the same two-path statement
- [x] Record in `.claude/notes/notes.md` that a per-frame Zarr group (`io/zarr/frame_io.rs:536,640`) and `MolRec::meta` (`core/store/record.rs:100`) carry meta untagged, so a dtype does not survive them. Making those paths tag-preserving is a separate 0.15 change, not this link and not a later minor.
- [ ] Run full check + test suite

## Testing strategy

Unit tests only, next to or mirroring the code under test (`.claude/notes/testing.md`). No end-to-end path, no external oracle, no source-text gate.

**Binder seam** — `molrs-python/tests/test_frame.py::TestFrameMeta`, one behaviour per test:

- Happy path: `f.meta["temperature"] = MetaValue("f32", 300.0)` then `f.meta["temperature"] = 310.0` → `dtype("temperature") == "f64"` and the value reads back `310.0`.
- Happy path: `MetaValue("f32", 300.0)` alone still yields `dtype("temperature") == "f32"` — the pin is the surviving door.
- Edge case: `f.meta["count"] = MetaValue("i64", 3)` then `f.meta["count"] = 1.5` succeeds (no `TypeError`) and reads `1.5`.
- Edge case: the read-write identity still holds for `i64` and `f64x6` keys, and explicitly no longer for `f32`.
- Unchanged, must stay green: `test_copying_a_frame_keeps_exact_dtypes` (`:234-242`) — `molrs.Frame(f)` goes through `PyFrameMeta`, not through a `dict`, so tags still copy whole.
- `setdefault` gets no new test: its early return at `:305` guarantees the key is absent, where `typed_for` already delegated to `infer_meta_value`, so the change there is refactor-only.

**Core durability** — `molrs/src/io/zarr/sequence.rs`, inside the existing test: a declared `i64` key receiving a frame `f64` is refused with an error naming both the key and `i64`. Hand-written inputs, no fixture corpus.

Green for the inner loop is `scripts/test-scope.sh` on the two touched modules; the closing gate is `cargo mrs-test && cargo mrs-doctest` plus the Python `tox -e py`.

**No regression example.** molrs has no `regressions/` tree: the benchmark and regression systems are being redesigned outside this repo (`CLAUDE.md` § Build & Test Commands), and the repo constitution outranks the generic template here. Precedent on file: `.claude/specs/cgsmiles-03-release.md:177`, and `.claude/specs/INDEX.md:27-29` records that a committed-corpus acceptance was already dropped from the 0.14 chain because the suites are unit-only. The role is carried by `TestFrameMeta`. The live docstring states the same rule the tests assert. This link does not edit a migration guide.

## Out of scope

- **Any behaviour change under `molrs/src/`.** The single hunk there is inside a `#[cfg(test)]` function; no symbol, signature or runtime path moves.
- **Making the frame-group / `MolRec` meta path tag-preserving.** Named in the Design and recorded in `.claude/notes/notes.md`. It is a stored-format change, separate from this binder rule. Compatibility is not the reason it stays out. If it is done, it is still 0.15, in its own change, not inside this link.
- **Deprecation shims.** Per `architecture-rules.md` § *Naming*, the rule is deleted, not deprecated. No `strict=` flag and no opt-in typed-slot mode.
- **Changing `mapping_to_meta_map`.** Its re-inference is cited as evidence the rule was never an invariant; after this link it is simply consistent with `__setitem__`.
- **A `Frame.to_bytes` / `Frame.from_bytes` method.** No such method exists on the Python `Frame` — the serde door is the module-level `molrs.io.write_frame_bytes` / `read_frame_bytes`. Adding methods is a separate public-API decision.
- **Migration guides, release notes, and version pins.** This link does not edit them. The two-path rule lives in the `PyFrameMeta` rustdoc and the `FrameMeta` docstring.
- **`molpy`, `molvis`, `molrec`.** The consumer sweep found none affected. This link does not wait on a tag to say so.
