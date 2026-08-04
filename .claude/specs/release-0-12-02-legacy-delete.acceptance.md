---
spec: release-0-12-02-legacy-delete
created: 2026-08-04
criteria:
  - id: ac-001
    summary: forcefield_method_json gone
    type: code
    pass_when: "No symbol forcefield_method_json or forcefield_meta module in tree"
    status: verified
    last_checked: 2026-08-04
  - id: ac-002
    summary: PairLJ126 kernels module gone
    type: code
    pass_when: "No pub mod kernels / PairLJ126 in molrs; PME tests use PairLJCut"
    status: verified
    last_checked: 2026-08-04
  - id: ac-003
    summary: read_trajectory_store gone
    type: code
    pass_when: "grep finds no read_trajectory_store definition or re-export"
    status: verified
    last_checked: 2026-08-04
  - id: ac-004
    summary: OPLSAATypifier.build removed
    type: runtime
    pass_when: "OPLSAATypifier has no build method in Rust or Python; tests use compose path"
    status: verified
    last_checked: 2026-08-04
  - id: ac-005
    summary: Python dual names removed
    type: code
    pass_when: "No PairLJ126Style* and no Entity/Entities aliases exported from molrs views/ff"
    status: verified
    last_checked: 2026-08-04
  - id: ac-006
    summary: suite green
    type: runtime
    pass_when: "cargo test -p molcrafts-molrs --lib --features full,filesystem and molrs-python tests pass"
    status: verified
    last_checked: 2026-08-04
out_of_scope:
  - Site documentation rewrite
  - Science math
---

# Acceptance — release-0-12-02-legacy-delete

All listed legacy public symbols are deleted with no shims; compose-path tests pass; suite green.
