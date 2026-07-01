---
slug: travis-parity-04-van-hove-and-reorientation
criteria:
  - id: ac-001
    summary: G_d(r,0) equals ρ·g(r) from the existing RDF
    type: scientific
    pass_when: |
      On a single frame, VanHove's distinct part at zero lag, G_d(r,0), equals the
      bulk density times compute::rdf's g(r) on the same configuration within the
      shared binning tolerance; G_s(r,0) is concentrated in the first r-bin.
    status: pending
  - id: ac-002
    summary: self Van Hove width tracks the MSD
    type: scientific
    pass_when: |
      For a trajectory with known diffusion, the second moment ∫ r² G_s(r,t) dr
      equals the MSD from compute::msd at each lag within 2%, and G_s(r,t) is
      Gaussian with width ∝ sqrt(t).
    status: pending
  - id: ac-003
    summary: Legendre C1/C2 match analytic rotation
    type: scientific
    pass_when: |
      A unit vector rotating at constant angular rate ω yields C_1(t) = cos(ωt) and
      C_2(t) = (3cos²(ωt) − 1)/2 within 1e-9; a static vector yields C_1 = C_2 = 1
      for all t.
    status: pending
  - id: ac-004
    summary: multi-origin averaging is stable
    type: code
    pass_when: |
      Van Hove and Legendre results change by less than a documented tolerance when
      the number of time origins is doubled (statistical convergence, not bias).
    status: pending
  - id: ac-005
    summary: reorientation is distinct from RotationalAutocorrelation
    type: code
    pass_when: |
      LegendreReorientation consumes a molecular vector (not a quaternion) and its
      rustdoc cross-references order::RotationalAutocorrelation, documenting that the
      two compute different observables; both remain independently usable.
    status: pending
  - id: ac-006
    summary: edge cases + full check green
    type: runtime
    pass_when: |
      A single-frame input yields only the t=0 result; a zero-length reorientation
      vector returns a typed ComputeError; module is WASM-clean and cargo fmt --check
      + clippy -D warnings + cargo test --features compute pass.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** are the two physical bridges that make Van Hove trustworthy:
  it must reduce to the RDF at t=0 and to the MSD in its second moment.
- **ac-003** is the analytic anchor for the Legendre reorientation TCFs.
- **ac-004** guards the multi-time-origin averaging both analyses rely on.
- **ac-005** prevents confusion with the pre-existing quaternion ACF — they are
  different observables and must stay clearly separated.
- **ac-006** covers degenerate inputs and the standard WASM-clean build gate.
