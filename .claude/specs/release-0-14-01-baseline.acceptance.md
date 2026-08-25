---
slug: release-0-14-01-baseline
created: 2026-08-25
criteria:
  - id: ac-001
    summary: every manifest reads 0.14.0 with no prerelease suffix
    type: code
    pass_when: |
      tomllib-parsing Cargo.toml, molrs-ffi/, molrs-wasm/, molrs-capi/,
      molrs-cxxapi/, molrs-python/Cargo.toml and molrs-python/pyproject.toml
      yields "0.14.0" for every version and every molcrafts-molrs /
      molcrafts-molrs-ffi pin; no string matches /\.dev|rc|a\d|b\d/.
    status: pending
  - id: ac-002
    summary: version parity is enforced by a test, not by grep
    type: runtime
    pass_when: |
      molrs-python/tests/test_version_parity.py exists, discovers manifests by
      globbing (not a hand-written name list), and passes under
      `uv --directory molrs-python run --no-sync tox -e py`.
    status: pending
  - id: ac-003
    summary: keys resolved to the published master Key shape
    type: code
    pass_when: |
      molrs/src/core/store/keys.rs and molrs-python/src/schema.rs contain no
      merge markers and expose the same Key constant set as origin/master;
      molrs-python/tests/test_ecs_pybind.py passes unmodified.
    status: pending
  - id: ac-004
    summary: release.md carries four backfilled records
    type: docs
    pass_when: |
      .claude/notes/release.md has "## v0.13.0", "## v0.13.1", "## v0.13.2"
      and "## v0.14.0" sections in the existing v0.12.1 format.
    status: pending
  - id: ac-005
    summary: merged tree is green on the default gate
    type: runtime
    pass_when: |
      `cargo test -p molcrafts-molrs --lib --features full,filesystem` and
      `cargo test --doc -p molcrafts-molrs --features full,filesystem` both
      pass on the merged dev branch.
    status: pending
  - id: ac-006
    summary: baseline regression checklist is committed
    type: runtime
    pass_when: |
      regressions/release-0-14-01-baseline.md lists all 8 version-string
      locations, the four release.md headings, and the keys ruling, and every
      listed check is reproducible by a reader with no access to this spec.
    status: pending
out_of_scope:
  - md subsystem code
  - tag / publish actions
  - any molpy change
---

# Acceptance — release-0-14-01-baseline

合并后的 `dev` 是 0.14.0 的唯一基线：版本串处处 `0.14.0`、keys 形状与已发布 master 一致、四条发布记录在案，且默认门与 doctest 门在这棵树上全绿。
