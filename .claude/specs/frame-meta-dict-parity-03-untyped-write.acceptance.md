---
spec: frame-meta-dict-parity-03-untyped-write
created: 2026-09-22
criteria:
  - id: ac-001
    summary: typed_for is gone and its four call sites infer
    type: code
    pass_when: |
      `PyFrameMeta::typed_for` does not exist in
      molrs-python/src/core/store/frame.rs, and `__setitem__`, both `absorb`
      arms and `setdefault` each call `infer_meta_value` directly. `tag_of`,
      `dtype`, `typed` and `PyMetaValue` still exist, and
      `meta_value_from_dtype` still has its two callers (`PyMetaValue::new`
      and the fixed-length-vector branch of `infer_meta_value`), so the
      deletion leaves no dead code and no new clippy warning.
    status: pending
  - id: ac-002
    summary: a plain write replaces an existing key's dtype
    type: runtime
    pass_when: |
      In molrs-python/tests/test_frame.py, after
      `f.meta["temperature"] = MetaValue("f32", 300.0)` and
      `f.meta["temperature"] = 310.0`, `f.meta.dtype("temperature") == "f64"`
      and `f.meta["temperature"] == pytest.approx(310.0)`; and
      `f.meta["count"] = MetaValue("i64", 3)` followed by
      `f.meta["count"] = 1.5` raises nothing and reads back 1.5.
    status: pending
  - id: ac-003
    summary: MetaValue still pins a dtype on every write path
    type: runtime
    pass_when: |
      `f.meta["temperature"] = MetaValue("f32", 300.0)` yields
      `dtype("temperature") == "f32"`, the same through `update()` and
      `setdefault()`, and test_copying_a_frame_keeps_exact_dtypes
      (molrs-python/tests/test_frame.py:234) passes unmodified.
    status: pending
  - id: ac-004
    summary: the declared-i64 vs frame-f64 refusal is asserted
    type: runtime
    pass_when: |
      `cargo mrs-test` passes and
      molrs/src/io/zarr/sequence.rs::a_meta_value_at_another_width_is_read_at_the_declared_one
      asserts an append error whose message contains both "count" and "i64"
      for a key declared i64 carrying MetaValue::F64(0.5).
    status: pending
  - id: ac-005
    summary: the only molrs/src hunk is inside the extended test fn
    type: code
    pass_when: |
      Every changed line under molrs/src/ in this link's diff falls inside the
      body of `a_meta_value_at_another_width_is_read_at_the_declared_one` in
      molrs/src/io/zarr/sequence.rs.
    status: pending
  - id: ac-006
    summary: the two durable paths are stated in the live docs
    type: docs
    pass_when: |
      The PyFrameMeta rustdoc (molrs-python/src/core/store/frame.rs) and the
      FrameMeta docstring (molrs-python/python/molrs/_lib.pyi) each name both
      a declared sequence schema and the serde/stream frame document, each
      carry the non-sticky qualifier ("dtype(k) reports the tag stored right
      now; any plain write re-infers it; MetaValue fixes the dtype of that
      write only"), and neither still claims a plain write keeps the key's
      dtype. This link does not edit migration-0-14.md or release.md.
    status: pending
  - id: ac-007
    summary: the untagged frame-group path has a durable home
    type: docs
    pass_when: |
      .claude/notes/notes.md carries a dated entry naming
      molrs/src/io/zarr/frame_io.rs:536,640 and
      molrs/src/core/store/record.rs:100 as paths that carry meta untagged.
      The entry does not defer the work past 0.15 for compatibility. Making
      those paths tag-preserving is a separate 0.15 change, not this link.
    status: pending
  - id: ac-008
    summary: full check and test suite green
    type: runtime
    pass_when: |
      `cargo fmt --check && cargo mrs-clippy -- -D warnings`,
      `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets --
      -D warnings`, `cargo mrs-test && cargo mrs-doctest`, and
      `uv --directory molrs-python run --no-sync tox -e py` all pass with no
      new warnings and no test skipped or weakened.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-005** are the two static reads. ac-005 is deliberately pinned to a named function rather than phrased as "no non-test hunk", because the latter needs a reviewer to classify a hunk before it can be checked.
- **ac-002 / ac-003** are the behaviour pair: the slot rule is gone, the `MetaValue` pin is not.
- **ac-004** closes the promise the existing test's doc comment already made. The test it extends already covers the silent-narrowing half; only refusal-by-width was uncovered.
- **ac-006 / ac-007** are the prose contract: the durability bound is the two live doc sites, and the untagged frame-group / MolRec paths are named without a compatibility deferral.
- **ac-008** is the closing gate.

There is no `regressions/` criterion: molrs has no such tree and the constitution routes regression work outside this repo (`.claude/specs/cgsmiles-03-release.md:177`; `.claude/specs/INDEX.md:27-29`).
