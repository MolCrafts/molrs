---
slug: release-0-12-05-cxxapi-panic-free
status: done
created: 2026-08-04
grilled: false
revised: 2026-08-04
depends_on:
  - release-0-12-01-harness
---

# release-0-12-05-cxxapi-panic-free — no panics across CXX bridge

## Summary

Every `extern "Rust"` / CXX-reachable function in `molrs-cxxapi` returns `Result` (or is infallible) and never panics on fallible I/O, schema insert, or empty stores. Atomiverse must not abort on bad paths.

## Domain basis

N/A (FFI safety). Rule: ffi.md — no panics in extern seams.

## Design

- Audit `molrs-cxxapi/src/lib.rs` for `.unwrap()`, `.expect()`, `panic!` on bridge-reachable paths.
- Convert non-`Result` exports that can fail: `write_frame_xyz`, zarr read/write, `xyz_read_first_frame`, `frame_set_column_*`, `frame_set_box`, helper builders `xyz_frame` / `frame_with_elements`.
- Map errors to `Result<T, String>` consistent with existing AM1-BCC / meta helpers.
- Rewrite tests that use `catch_unwind` to assert `Err`.
- Leave chemistry logic unchanged.

### Reuse decision

- `reuse` existing `Result<(), String>` helpers (`write_xyz_path`, `frame_set_meta_entry`, AM1-BCC).
- `pattern` from those helpers for all remaining panicking entries.
- `new` — none.

## Files to create or modify

- `molrs-cxxapi/src/lib.rs`
- `molrs-cxxapi/src/bridge.rs` (if signatures change)
- `molrs-cxxapi/tests/*.rs` as needed

## Tasks

- [x] Write failing tests asserting Result errors for missing file / empty zarr / bad insert (no catch_unwind)
- [x] Convert I/O CXX exports to Result (xyz, zarr)
- [x] Convert frame mutators (`frame_set_column_*`, `frame_set_box`) to Result
- [x] Convert fallible helpers used by bridge; remove residual expect on AM1-BCC insert
- [x] Grep-gate: no unwrap/expect/panic outside `#[cfg(test)]` on bridge bodies (spot-check remaining analysis helpers)
- [x] Add regression `regressions/release-0-12-05-cxxapi-panic-free.md` listing converted exports
- [x] Run full check + test suite (`cargo test -p molcrafts-molrs-cxxapi`)

## Testing strategy

Unit/integration tests in cxxapi crate only. Bad path → `Err(String)` containing actionable message.

## Out of scope

- C API U8 dtype polish (nice-to-have)
- WASM RDF view unwrap (nice-to-have)
- Python sys.modules registration for schema
