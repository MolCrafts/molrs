---
slug: release-0-14-13-frame-store-naming
created: 2026-08-25
criteria:
  - id: ac-001
    summary: no public identifier spells MolRec, molrec or zarr
    type: code
    pass_when: |
      A predicate test scanning molrs, molrs-cxxapi, molrs-python, molrs-ffi,
      molrs-wasm, molrs-capi source trees plus README / site-src finds zero
      public identifiers or user-visible doc lines containing MolRec, molrec or
      zarr, except three exemptions asserted inline in the test body:
      molrs/src/core/store/record.rs, the io::store::zarr adapter module path,
      and provenance URLs pointing at the external molrec contract.
    status: pending
  - id: ac-002
    summary: the record aggregate is exported as Record with read / write
    type: runtime
    pass_when: |
      molrs.Record exists with a static read(path) and an instance write(path);
      hasattr(molrs, "MolRec") is False and neither Record nor Trajectory
      exposes read_zarr or write_zarr.
    status: pending
  - id: ac-003
    summary: the Rust crate root exposes Record, and record.rs is untouched
    type: code
    pass_when: |
      molrs/src/core/mod.rs re-exports the aggregate as molrs::Record and
      `git diff` shows no change to molrs/src/core/store/record.rs.
    status: pending
  - id: ac-004
    summary: the cxxapi bridge names the object, not the backend
    type: code
    pass_when: |
      molrs-cxxapi declares write_frame and read_first_frame in bridge.rs,
      lib.rs and build.rs; write_frame_zarr and read_frame_zarr_first appear
      nowhere, and no deprecated alias was left behind.
    status: pending
  - id: ac-005
    summary: the external consumer and the naming principle are on record
    type: docs
    pass_when: |
      .claude/notes/notes.md carries a dated entry stating the naming principle
      (public APIs named by object, never by an undecided backend) and the
      old→new cxxapi mapping together with the Atomiverse consumer
      (cpu::ZarrReader) it breaks.
    status: pending
  - id: ac-006
    summary: renaming moved no bytes on disk
    type: runtime
    pass_when: |
      A record written before the rename still loads through Record.read, and a
      record written after it carries meta["format_name"] == "molrec"
      unchanged; round-tripped coordinate arrays compare bit-identical
      (assert_array_equal, no tolerance).
    status: pending
  - id: ac-007
    summary: the phantom MolRecReader is gone
    type: docs
    pass_when: |
      molrs-wasm/README.md no longer lists MolRecReader, a symbol that never
      existed in molrs-wasm/src.
    status: pending
  - id: ac-008
    summary: naming regression reproduces its hard-coded golden
    type: runtime
    pass_when: |
      `python regressions/release-0-14-13-frame-store-naming.py` exits 0,
      imports no third-party scientific package, round-trips a 3-atom Record to
      its embedded coordinate goldens bitwise, and asserts zero hits for the old
      names in dir(molrs).
    status: pending
  - id: ac-009
    summary: full gate green after the rename
    type: runtime
    pass_when: |
      cargo lib tests, cargo doc tests, the cxxapi build and the molrs-python
      tox py env all pass.
    status: pending
out_of_scope:
  - internal core/store/record.rs machinery and the separate molrec test suite
  - migration-guide entries for the renames (spec 06)
  - Atomiverse-side follow-up for cpu::ZarrReader
  - full removal of the zarr UnitSystem enum
  - the io::store::zarr adapter module path
---

# Acceptance — release-0-14-13-frame-store-naming

公开面上再没有 `MolRec` / `zarr` 这两个词：名字说的是对象（record / frame），技术名只活在适配层与出处引用里；内部机制与落盘契约一个字节没动，外部消费者的破坏被写在案上而不是留给下游自己撞。
