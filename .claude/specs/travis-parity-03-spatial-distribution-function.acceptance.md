---
slug: travis-parity-03-spatial-distribution-function
criteria:
  - id: ac-001
    summary: native Kabsch recovers a known rigid rotation
    type: scientific
    pass_when: |
      Given a template and a copy rotated by a known R and translated by t,
      kabsch() returns a rotation with det = +1 matching R within 1e-9 and RMSD
      below 1e-9; a chiral template confirms the reflection guard never returns a
      det = −1 improper rotation.
    status: pending
  - id: ac-002
    summary: SDF is frame-invariant (the defining property)
    type: scientific
    pass_when: |
      A target atom held at a fixed offset in the reference molecule's body frame,
      while the whole assembly tumbles arbitrarily each frame, accumulates into a
      single sharp density voxel at that offset — whereas the same data through
      lab-frame GaussianDensity smears across a spherical shell.
    status: pending
  - id: ac-003
    summary: bulk-normalized g_SDF approaches 1 far from the reference
    type: code
    pass_when: |
      For an ideal-gas target distribution, the bulk-normalized g_SDF tends to 1
      (within statistical tolerance) in voxels far from the reference COM.
    status: pending
  - id: ac-004
    summary: target unwrapping honors PBC about the reference COM
    type: code
    pass_when: |
      A target atom on the opposite side of a periodic boundary from the reference
      COM is binned at its minimum-image position relative to that COM.
    status: pending
  - id: ac-005
    summary: orientation field behaves correctly
    type: code
    pass_when: |
      A target unit vector fixed in the reference frame yields a per-voxel mean
      vector equal to it (norm ≈ 1); a uniformly random target vector yields a mean
      vector with norm → 0.
    status: pending
  - id: ac-006
    summary: degenerate reference rejected; full check green
    type: runtime
    pass_when: |
      A reference set with fewer than 3 non-collinear atoms returns a typed
      ComputeError; the module is BLAS-free/WASM-clean; and cargo fmt --check +
      clippy -D warnings + cargo test --features compute pass.
    status: pending
---

# Acceptance criteria

- **ac-001** validates the native, BLAS-free Kabsch (with the proper-rotation
  reflection guard) that the whole SDF rests on.
- **ac-002** is the headline scientific property — frame invariance is exactly what
  separates an SDF from molrs's existing lab-frame density.
- **ac-003 / ac-004 / ac-005** pin bulk normalization, PBC unwrapping, and the
  orientation map.
- **ac-006** enforces fail-fast on a degenerate reference and the standard
  WASM-clean build gate.
