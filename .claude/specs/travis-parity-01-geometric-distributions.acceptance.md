---
slug: travis-parity-01-geometric-distributions
criteria:
  - id: ac-001
    summary: AngleObservable + DistributionFunction reproduce a known ADF
    type: scientific
    pass_when: |
      Building an AngleObservable over a frame whose triples form 60° angles and
      histogramming it with DistributionFunction yields a normalized density whose
      peak bin contains 60° (±1 bin), with f64 agreement to a hand-computed angle
      within 1e-9 before binning.
    status: pending
  - id: ac-002
    summary: DihedralObservable matches IUPAC-signed reference torsions
    type: scientific
    pass_when: |
      For fixed four-atom configurations with known torsions (0°, +90°, 180°,
      −90°), DihedralObservable returns the IUPAC-signed φ in (−π, π] matching the
      analytic value within 1e-9.
    status: pending
  - id: ac-003
    summary: distance DF over explicit pairs honors the minimum image under PBC
    type: code
    pass_when: |
      A DistanceObservable over a pair straddling a periodic boundary returns the
      minimum-image distance (matching compute::rdf for the same pair), not the
      raw in-box separation.
    status: pending
  - id: ac-004
    summary: histograms are correctly normalized
    type: code
    pass_when: |
      For each distribution function, the trapezoidal integral of the normalized
      density over its bin range equals 1 within 1e-6, and the raw counts sum to
      n_samples × n_frames.
    status: pending
  - id: ac-005
    summary: raw vs sin θ-corrected ADF both available and distinct
    type: code
    pass_when: |
      DistributionFunction exposes both the raw ADF density and the sin θ-corrected
      ADF density; for a non-uniform angular sample the two differ, and the
      corrected one is the raw divided by sin θ then renormalized.
    status: pending
  - id: ac-006
    summary: degenerate geometry does not produce NaN or panic
    type: code
    pass_when: |
      A collinear triple yields angle 0 or π (no NaN); a zero-length vector yields a
      typed ComputeError (not a silent NaN); an empty AtomGroups yields an empty
      DistributionResult without panicking.
    status: pending
  - id: ac-007
    summary: module gated on compute, WASM-clean, full check green
    type: runtime
    pass_when: |
      compute/distribution compiles only under the `compute` feature, pulls in no
      BLAS/FFI/new crate, and cargo fmt --check + clippy -D warnings +
      cargo test --features compute all pass.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** are the scientific anchors — the angle and dihedral
  observables must reproduce closed-form geometry, since every downstream
  distribution (and the link-02 CDF) trusts these samples.
- **ac-003** locks the PBC contract to the existing `compute::rdf` minimum-image
  behavior so distance DFs and RDFs agree on the same pair.
- **ac-004 / ac-005** pin the normalization and the raw-vs-`sinθ` ADF convention
  that TRAVIS users expect and that is the most common source of ADF confusion.
- **ac-006** is the robustness gate (collinear / zero-length / empty), following
  molrs's edge-case testing standard.
- **ac-007** enforces the feature gating and the WASM-clean / no-new-dependency
  constraint, and is the standard build gate.
