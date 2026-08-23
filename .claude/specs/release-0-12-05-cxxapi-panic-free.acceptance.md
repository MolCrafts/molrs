---
spec: release-0-12-05-cxxapi-panic-free
created: 2026-08-04
criteria:
  - id: ac-001
    summary: I/O CXX exports do not panic
    type: runtime
    pass_when: "write/read xyz/zarr bridge functions return Err on bad path/empty store without abort"
    status: verified
    last_checked: 2026-08-04
  - id: ac-002
    summary: frame mutators do not panic
    type: runtime
    pass_when: "frame_set_column_* and frame_set_box return Result and never unwrap inserts"
    status: verified
    last_checked: 2026-08-04
  - id: ac-003
    summary: no unwrap/expect on bridge bodies
    type: code
    pass_when: "rg for unwrap/expect/panic in molrs-cxxapi/src finds only cfg(test) allowlist entries"
    status: verified
    last_checked: 2026-08-04
  - id: ac-004
    summary: cxxapi tests green
    type: runtime
    pass_when: "cargo test --manifest-path molrs-cxxapi/Cargo.toml passes"
    status: verified
    last_checked: 2026-08-04
out_of_scope:
  - capi/wasm non-blocking polish
---

# Acceptance — release-0-12-05-cxxapi-panic-free

CXX bridge is panic-free on all fallible paths; errors surface as Result.
