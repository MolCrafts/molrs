---
spec: release-0-12-04-science-dielectric-zarr
created: 2026-08-04
criteria:
  - id: ac-001
    summary: components dielectric uses two-pass variance
    type: scientific
    pass_when: "static_dielectric_constant_components matches isotropic mean on large-offset synthetic dipoles where one-pass would fail"
    status: verified
    last_checked: 2026-08-04
  - id: ac-002
    summary: analysis time documented as fs in compute
    type: docs
    pass_when: "dielectric and conductivity rustdocs use fs; no ps unit table for analysis dt"
    status: verified
    last_checked: 2026-08-04
  - id: ac-003
    summary: SI prefactors use 1e-15 for time
    type: scientific
    pass_when: "conductivity SI paths convert fs→s with 1e-15 (not 1e-12)"
    status: verified
    last_checked: 2026-08-04
  - id: ac-004
    summary: SimBox zarr is f64
    type: runtime
    pass_when: "write/read SimBox preserves f64 ULP; legacy f32 arrays still load"
    status: verified
    last_checked: 2026-08-04
  - id: ac-005
    summary: suite green
    type: runtime
    pass_when: "cargo test -p molcrafts-molrs --lib --features full,filesystem passes"
    status: verified
    last_checked: 2026-08-04
out_of_scope:
  - VACF
  - molpy docs
---

# Acceptance — release-0-12-04-science-dielectric-zarr

Dielectric/conductivity analysis is fs-consistent; per-axis ε is numerically stable; SimBox geometry survives Zarr at f64.
