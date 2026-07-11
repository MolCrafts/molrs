---
slug: graph-sink-04-hydrogens-coords
created: 2026-07-11
criteria:
  - id: ac-001
    summary: H gains xyz when heavy has xyz
    type: scientific
    pass_when: |
      Bare carbon at (0,0,0) after add_hydrogens has 4 H; each H has x,y,z;
      distance(C,H) within 1.09 ± 0.02 Å for all four.
    status: verified
    last_checked: 2026-07-11
    evidence: test_add_hydrogens_places_coords_when_heavy_has_xyz
  - id: ac-002
    summary: bond-order valence counts unchanged (ethene 4H not 6H)
    type: scientific
    pass_when: |
      C=C (order 2.0) → exactly 4 hydrogens added total (existing chem tests still pass).
    status: verified
    last_checked: 2026-07-11
    evidence: test_ethylene_c_double_c + full chem::hydrogens suite
  - id: ac-003
    summary: no xyz on heavy → H without coordinates
    type: code
    pass_when: |
      Bare C without x,y,z → H atoms exist; has(h,"x") is false for each new H.
    status: verified
    last_checked: 2026-07-11
    evidence: test_add_hydrogens_no_xyz_when_heavy_lacks_coords
  - id: ac-004
    summary: parent higher-order relations retained
    type: code
    pass_when: |
      Ethane with generate_topology angles; add_hydrogens does not drop those angles
      (n_angles after >= n_angles before on the original heavy skeleton).
    status: verified
    last_checked: 2026-07-11
    evidence: test_add_hydrogens_preserves_parent_angles
  - id: ac-005
    summary: quality gate
    type: runtime
    pass_when: |
      cargo fmt/clippy/test --all-features exit 0.
    status: verified
    last_checked: 2026-07-11
out_of_scope:
  - molpy complete_valence wrapper
  - geometry optimization
---

# Acceptance — graph-sink-04-hydrogens-coords

Done means `add_hydrogens` is sufficient for typify-region capping: correct H counts and
usable 3D positions when the fragment already has coordinates.
