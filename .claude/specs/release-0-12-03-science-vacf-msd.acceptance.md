---
spec: release-0-12-03-science-vacf-msd
created: 2026-08-04
criteria:
  - id: ac-001
    summary: VACF uses unbiased /(n-τ)
    type: scientific
    pass_when: "velocity_acf divides by (n-τ); unit test matches direct O(n²) reference within 1e-10 relative"
    status: verified
    last_checked: 2026-08-04
  - id: ac-002
    summary: GreenKuboDiffusion shares VACF curve
    type: runtime
    pass_when: "GreenKuboDiffusion.acf equals VACF.acf on same input; docs state D=1/d ∫C"
    status: verified
    last_checked: 2026-08-04
  - id: ac-003
    summary: VACFAccumulator matches batch
    type: runtime
    pass_when: "finalize() equals batch VACF on identical series"
    status: verified
    last_checked: 2026-08-04
  - id: ac-004
    summary: MSD unwrap precondition documented and tested
    type: scientific
    pass_when: "MSD/Einstein rustdoc requires unwrapped positions; wrapped vs unwrapped regression exists and shows saturation vs quadratic"
    status: verified
    last_checked: 2026-08-04
  - id: ac-005
    summary: suite green
    type: runtime
    pass_when: "cargo test -p molcrafts-molrs --lib --features full,filesystem passes"
    status: verified
    last_checked: 2026-08-04
out_of_scope:
  - Dielectric/conductivity SI unit migration
  - Zarr dtype
---

# Acceptance — release-0-12-03-science-vacf-msd

VACF is the unbiased scientific ACF; Green–Kubo D docs match math; MSD unwrap contract is explicit and tested.
