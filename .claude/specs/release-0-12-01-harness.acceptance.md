---
spec: release-0-12-01-harness
created: 2026-08-04
criteria:
  - id: ac-001
    summary: architecture.md is a real single-crate inventory
    type: docs
    pass_when: ".claude/notes/architecture.md lists molrs/src modules and binders; no empty map stub"
    status: verified
    last_checked: 2026-08-04
  - id: ac-002
    summary: architecture-rules describe single-crate graph
    type: docs
    pass_when: "architecture-rules.md has no multi-crate table claiming molrs-core/io/ff as separate crates"
    status: verified
    last_checked: 2026-08-04
  - id: ac-003
    summary: ffi.md lists four binders
    type: docs
    pass_when: "ffi.md names molrs-python, molrs-wasm, molrs-capi, molrs-cxxapi and molrs-ffi"
    status: verified
    last_checked: 2026-08-04
  - id: ac-004
    summary: science.md analysis time is fs
    type: docs
    pass_when: "science.md unit table Time = fs with no ps dual for analysis"
    status: verified
    last_checked: 2026-08-04
  - id: ac-005
    summary: CLAUDE.md layer lies fixed
    type: docs
    pass_when: "CLAUDE.md does not claim optimize always-on, Gasteiger under perceive, or SMARTS under core/chem"
    status: verified
    last_checked: 2026-08-04
out_of_scope:
  - Code kernel science fixes
  - Public API deletion
---

# Acceptance — release-0-12-01-harness

Harness notes and CLAUDE.md accurately describe the 0.12 single-crate layout and fs analysis unit system so later release specs can be implemented without agent mis-navigation.
