---
slug: travis-parity-06-radical-voronoi
criteria:
  - id: ac-001
    summary: total cell volume equals the box volume
    type: scientific
    pass_when: |
      For a random periodic configuration with per-atom radii, the sum of all
      radical-Voronoi cell volumes equals the SimBox volume within 1e-9 relative
      error.
    status: pending
  - id: ac-002
    summary: analytic cells and the radical plane are correct
    type: scientific
    pass_when: |
      An equal-radius simple-cubic lattice yields unit-cube cells (volume + 6 faces);
      a two-atom unequal-radius case places the shared face at the analytic radical
      plane |x−xi|²−Ri² = |x−xj|²−Rj²; equal radii reproduce the plain Voronoi
      bisector.
    status: pending
  - id: ac-003
    summary: tessellation is periodic and neighbor-symmetric
    type: code
    pass_when: |
      Cells wrap across periodic boundaries; the neighbor relation is symmetric
      (i has j as a face-neighbor iff j has i), and the shared face areas agree
      within 1e-9.
    status: pending
  - id: ac-004
    summary: domain analysis recovers constructed domains
    type: code
    pass_when: |
      A constructed two-label bilayer yields exactly two domains of the expected
      sizes; an interpenetrating mixture yields a single percolating domain — via
      union-find over the cell-adjacency graph.
    status: pending
  - id: ac-005
    summary: void analysis recovers a constructed cavity
    type: code
    pass_when: |
      A lattice with one atom removed yields a single cavity whose volume matches the
      removed cell's volume within tolerance, and a total void fraction consistent
      with the construction.
    status: pending
  - id: ac-006
    summary: native backend is WASM-clean; full check green
    type: runtime
    pass_when: |
      The default `voronoi` feature builds for wasm32 with no C/C++ FFI dependency;
      any voro_rs oracle is behind a separate non-default feature; cargo fmt --check
      + clippy -D warnings + cargo test --features voronoi pass.
    status: pending
---

# Acceptance criteria

- **ac-001** is the decisive correctness check — a radical tessellation that does not
  conserve volume is wrong.
- **ac-002** validates the radical-plane geometry (and that it degrades to plain
  Voronoi at equal radii).
- **ac-003** pins periodicity and the face/neighbor symmetry the consumers rely on.
- **ac-004 / ac-005** justify shipping the core by exercising its two real consumers
  (domain + void), satisfying the no-speculative-code rule.
- **ac-006** enforces the WASM-clean native backend (the whole reason to reject C++
  FFI as the default) and the standard build gate.
