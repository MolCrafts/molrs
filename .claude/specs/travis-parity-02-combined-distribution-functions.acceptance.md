---
slug: travis-parity-02-combined-distribution-functions
criteria:
  - id: ac-001
    summary: 2-D CDF marginals equal the link-01 1-D distributions
    type: scientific
    pass_when: |
      For a distance×angle CombinedDistribution, summing the normalized joint
      density over the angle axis reproduces the link-01 distance DistributionResult
      density, and over the distance axis reproduces the ADF density, each within
      1e-6 per bin (after matching bin grids).
    status: pending
  - id: ac-002
    summary: correlation vs independence is resolved
    type: code
    pass_when: |
      A synthetic dataset where angle is a deterministic function of distance yields
      joint density confined to a diagonal band (off-band bins ≈ 0); a synthetic
      independent dataset yields density equal to the outer product of its marginals
      within 1e-6.
    status: pending
  - id: ac-003
    summary: joint density is normalized
    type: code
    pass_when: |
      The discrete integral of the normalized joint density over all axes equals 1
      within 1e-6 for both the 2-D and 3-D cases.
    status: pending
  - id: ac-004
    summary: mismatched observable sample counts are rejected
    type: code
    pass_when: |
      Constructing a CombinedDistribution whose observables emit different numbers
      of per-frame samples returns a typed ComputeError on compute rather than
      zipping to the shorter length.
    status: pending
  - id: ac-005
    summary: free_energy is finite and floored
    type: code
    pass_when: |
      free_energy(T) returns −kT·ln p (kB = 1.987204e-3 kcal/(mol·K)) that is finite
      on populated bins and equals the documented floor (not −inf/NaN) on empty bins.
    status: pending
  - id: ac-006
    summary: full check green, WASM-clean
    type: runtime
    pass_when: |
      compute/distribution/combined compiles under the `compute` feature with no new
      crate/BLAS/FFI, and cargo fmt --check + clippy -D warnings +
      cargo test --features compute pass.
    status: pending
---

# Acceptance criteria

- **ac-001** is the defining contract: a CDF must be consistent with its link-01
  marginals, or it is not a joint distribution of the same observables.
- **ac-002** proves the CDF actually captures correlation (its entire reason to
  exist) and degrades to a product under independence.
- **ac-003 / ac-005** pin normalization and the free-energy floor (no −∞/NaN).
- **ac-004** enforces fail-fast validation over silent zip-truncation, per the
  project's input-validation rule.
- **ac-006** is the standard build + WASM-clean + no-new-dependency gate.
