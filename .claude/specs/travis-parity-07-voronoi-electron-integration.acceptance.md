---
slug: travis-parity-07-voronoi-electron-integration
criteria:
  - id: ac-001
    summary: Cube trajectory reader iterates real fixtures
    type: code
    pass_when: |
      CubeTrajectory reads a multi-frame fixture set under tests-data/cube_traj/,
      iterating every file (never synthetic), yielding the correct frame count and a
      consistent grid shape + atom list per frame, with Bohr→Å normalization applied
      at the boundary.
    status: pending
  - id: ac-002
    summary: integrated electronic charge is conserved
    type: scientific
    pass_when: |
      The sum over all radical-Voronoi cells of the integrated electronic charge
      equals the grid total ∫ρ dV within the grid discretization tolerance, and the
      per-molecule total charge sums to the system charge.
    status: pending
  - id: ac-003
    summary: integrated dipole matches an analytic distribution
    type: scientific
    pass_when: |
      A constructed grid charge distribution integrates (nuclei + electronic) to its
      analytic dipole within grid tolerance; a centrosymmetric density yields a
      dipole magnitude near zero; origin-dependence for a charged species is
      documented and exercised.
    status: pending
  - id: ac-004
    summary: unit conversion is correct
    type: code
    pass_when: |
      Bohr→Å and e/Bohr³→e conversions are applied so that a hand-computed reference
      moment in molrs units (Å, e) is reproduced within 1e-9.
    status: pending
  - id: ac-005
    summary: finite-field polarizability recovers a linear-response input
    type: scientific
    pass_when: |
      Given synthetic moment sets generated from a known α tensor under ±E fields,
      polarizability_finite_field recovers that α within the central-difference
      truncation tolerance.
    status: pending
  - id: ac-006
    summary: boundary/PBC robustness + full check green
    type: runtime
    pass_when: |
      Voxels on a cell boundary are assigned deterministically; a molecule split
      across the periodic boundary is integrated via min-image; cargo fmt --check +
      clippy -D warnings + cargo test --features voronoi pass.
    status: pending
---

# Acceptance criteria

- **ac-001** enforces the mandatory IO testing rule (real multi-frame cube fixtures,
  iterate every file) for the new trajectory reader.
- **ac-002 / ac-003** are the physics anchors — charge conservation and a correct
  analytic dipole are what make the downstream spectra meaningful.
- **ac-004** pins the atomic-unit→molrs-unit conversion at the reader boundary.
- **ac-005** validates the finite-field polarizability that link 08's Raman/ROA need.
- **ac-006** covers boundary-voxel determinism, PBC molecule handling, and the build
  gate.
