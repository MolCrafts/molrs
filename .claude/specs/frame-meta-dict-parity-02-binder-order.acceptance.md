---
spec: frame-meta-dict-parity-02-binder-order
created: 2026-09-22
criteria:
  - id: ac-001
    summary: Link 01 has landed; its wire and Zarr guarantees are in place
    type: code
    pass_when: |
      Verified by inspection: molrs/src/core/store/meta.rs MetaMap is backed by
      an insertion-ordered map (not std HashMap); molrs/src/serialize.rs uses
      IndexMap in BOTH impl Serialize for Frame (~:511) and FrameRepr (~:527);
      molrs/src/io/zarr/frame_io.rs carries link 01's frame-group meta order
      round-trip test. If any of the three is absent, this link is blocked.
    status: pending
  - id: ac-002
    summary: PyFrameMeta::keys performs no sort
    type: code
    pass_when: |
      molrs-python/src/core/store/frame.rs fn keys contains no sort /
      sort_unstable / sort_by call and returns the map's own key order; no new
      public symbol is added to the file.
    status: pending
  - id: ac-003
    summary: Python FrameMeta order tests pass under tox
    type: runtime
    pass_when: |
      `uv --directory molrs-python run --no-sync tox -e py` is green, and
      tests/test_frame.py::TestFrameMeta contains a case whose fixture inserts
      keys in non-alphabetical order and asserts list(f.meta), f.meta.keys(),
      f.meta.items(), f.meta.values(), dict(f.meta) and repr(f.meta) all equal
      that insertion order.
    status: pending
  - id: ac-004
    summary: popitem returns the last-inserted key (dict LIFO parity)
    type: runtime
    pass_when: |
      A test in tests/test_frame.py asserts that after inserting three keys in
      non-alphabetical order, f.meta.popitem() returns the LAST-INSERTED key,
      not the lexicographically last one, and it passes under tox -e py.
    status: pending
  - id: ac-005
    summary: test_mapping_protocol's fixture can distinguish order
    type: code
    pass_when: |
      molrs-python/tests/test_frame.py test_mapping_protocol no longer uses
      {"a": 1, "b": "two"}: its keys are inserted in non-alphabetical order and
      the assertion is on list(f.meta), not sorted(f.meta).
    status: pending
  - id: ac-006
    summary: capi order assertion lives in the C++ file and ctest runs it green
    type: runtime
    pass_when: |
      molrs-capi/tests/cpp/test_molrs_capi.cpp contains TEST_F(MolrsTest,
      FrameMetadataOrder) exercising molrs_frame_meta_count and
      molrs_frame_meta_key over a non-alphabetical insertion sequence plus an
      out-of-range index, AND `cargo build --manifest-path
      molrs-capi/Cargo.toml && cmake -S molrs-capi/tests/cpp -B
      molrs-capi/build-test && cmake --build molrs-capi/build-test && ctest
      --test-dir molrs-capi/build-test --output-on-failure` is green. No
      #[cfg(test)] mod tests is added to molrs-capi/src/frame.rs.
    status: pending
  - id: ac-007
    summary: molrs_frame_meta_key drops its sort and its stale rustdoc
    type: code
    pass_when: |
      molrs-capi/src/frame.rs molrs_frame_meta_key contains no sort call, and
      its doc comment (was :858 "lexicographically sorted") states insertion
      order instead. molrs-capi/include/molrs.h regenerates byte-identical
      (cbindgen.toml:7 documentation = false); a header diff is a failure.
    status: pending
  - id: ac-008
    summary: frame_meta_entries drops its sort and its order test is green
    type: runtime
    pass_when: |
      molrs-cxxapi/src/lib.rs frame_meta_entries contains no sort_by_key, and
      `cargo test --manifest-path molrs-cxxapi/Cargo.toml` is green with a new
      frame_meta_entries_follow_insertion_order test asserting the full key
      sequence from a non-alphabetical insertion order.
    status: pending
  - id: ac-009
    summary: wasm metaNames test asserts sequence equality (CI-only gate)
    type: runtime
    pass_when: |
      molrs-wasm/src/core/frame.rs's meta_names test asserts Vec equality
      against frame_with_meta's insertion order (["energy", "config"]) rather
      than len + contains, and `wasm-pack test --node` in molrs-wasm is green.
      Note for the verifier: the pre-push wasm hook only BUILDS, so this bar is
      met by ci-wasm.yml:69 or by running wasm-pack test --node by hand.
    status: pending
  - id: ac-010
    summary: Order-bearing docs updated; scope wording kept out of rustdoc
    type: docs
    pass_when: |
      The metaNames rustdoc (molrs-wasm/src/core/frame.rs) and the FrameMeta
      docstring (molrs-python/python/molrs/_lib.pyi) each state that
      enumeration follows insertion order; no rustdoc on MetaMap
      (molrs/src/core/store/meta.rs) or write_frame_group
      (molrs/src/io/zarr/frame_io.rs) contains this spec's scope wording
      (e.g. "incidental", "not guaranteed") about Zarr order.
    status: pending
  - id: ac-011
    summary: All four binder enumeration surfaces report one order
    type: code
    pass_when: |
      By inspection, each of PyFrameMeta::keys
      (molrs-python/src/core/store/frame.rs), Frame::meta_names
      (molrs-wasm/src/core/frame.rs), molrs_frame_meta_key
      (molrs-capi/src/frame.rs) and frame_meta_entries
      (molrs-cxxapi/src/lib.rs) enumerates frame.meta with no re-ordering step,
      and each has a passing order test in its own gate (ac-003, ac-009,
      ac-006, ac-008). Scope: this claim covers these four binder surfaces; the
      serde wire is link 01's, guaranteed via ac-001.
    status: pending
  - id: ac-012
    summary: Found debt recorded with path:line and routes
    type: docs
    pass_when: |
      .claude/notes/notes.md carries a dated entry naming (1) the dead
      molrs-capi Rust suite with molrs-capi/src/schema.rs:150-168 as evidence
      and "add cargo test --manifest-path molrs-capi/Cargo.toml to
      .pre-commit-config.yaml capi-tests and .github/workflows/ci-capi.yml" as
      the /mol:fix route, (2) capi's clone_frame-per-metadata-read with
      molrs-ffi/src/store.rs:117 Store::with_frame as the /mol:refactor target,
      and (3) PyFrameMeta::map's clone-per-read routed to link 04.
    status: pending
  - id: ac-013
    summary: Full check and test suite green
    type: runtime
    pass_when: |
      `cargo fmt --check && cargo mrs-clippy -- -D warnings && cargo clippy
      --manifest-path molrs-cxxapi/Cargo.toml --all-targets -- -D warnings`
      passes, `cargo mrs-test && cargo mrs-doctest` passes, and `prek run
      --all-files --hook-stage pre-push` passes. The wasm order assertion is
      covered by ac-009, which that hook does not execute.
    status: pending
---

# Acceptance criteria

**ac-001** is a precondition, not work: this link is meaningless if 01 has not landed, and it is verified by reading three files rather than by running anything.

**ac-002 / ac-007 / ac-008** are the three deletions. Each is a source-inspection bar because the absence of a sort is what is being asserted; the behavioural consequence is asserted by the matching runtime criterion.

**ac-006** is the criterion the previous draft got wrong. The assertion must live in `molrs-capi/tests/cpp/test_molrs_capi.cpp` because no gate in this repository runs `cargo test --manifest-path molrs-capi/Cargo.toml`, which makes `#[cfg(test)]` in `molrs-capi/src/**` compile-only. Shipping a breaking C ABI semantic change behind a test that never executes is what this criterion exists to prevent.

**ac-009** carries its own gate warning: a green local pre-push run has not executed it.

**ac-011** states its own bound. Five surfaces enumerate metadata in this repository; four are this link's and the fifth (the serde wire) is link 01's, reached through `ac-001`.

**ac-012** is the Iron-law bar: three pieces of rot were found while drafting, none is fixed here, and all three must be written down with `path:line` and a route before this link can close.

There is no `regressions/` criterion: molrs has no such tree and the constitution routes regression work outside this repo (`.claude/specs/cgsmiles-03-release.md:177`).
