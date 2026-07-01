---
slug: travis-parity-05-hydrogen-bond-network
criteria:
  - id: ac-001
    summary: geometric detection matches a known dimer + cutoff boundaries
    type: scientific
    pass_when: |
      A water dimer at a geometry inside the (distance, angle) criterion is detected
      as exactly one D–H···A bond; perturbing the angle just past the cutoff (or the
      distance past the cutoff) drops the bond; the reported D···A distance and
      D–H···A angle match hand-computed values within 1e-9.
    status: pending
  - id: ac-002
    summary: detection honors the minimum image under PBC
    type: code
    pass_when: |
      A donor/acceptor pair separated across a periodic boundary is detected using
      the minimum-image distance and angle, consistent with compute neighbor search.
    status: pending
  - id: ac-003
    summary: network components match a constructed topology
    type: code
    pass_when: |
      A chain of N H-bonded molecules yields a single connected component of size N;
      removing the central bond yields two components of the expected sizes — all via
      core::Topology.
    status: pending
  - id: ac-004
    summary: no petgraph in the compute hbond path
    type: code
    pass_when: |
      A source/AST scan of compute/hbond finds no use of petgraph; the module builds
      with the `compute` feature alone (the `smiles` feature is NOT required).
    status: pending
  - id: ac-005
    summary: continuous vs intermittent lifetimes are correct
    type: scientific
    pass_when: |
      For a synthetic binary bond-presence series, the continuous S_HB(t) and
      intermittent C_HB(t) match the analytic correlation functions, and the
      intermittent lifetime τ ≥ the continuous lifetime.
    status: pending
  - id: ac-006
    summary: edge cases + full check green
    type: runtime
    pass_when: |
      No donors/acceptors → empty result (no panic); self-bonds excluded and no
      double counting; cargo fmt --check + clippy -D warnings +
      cargo test --features compute pass.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** lock the geometric criterion (Luzar–Chandler defaults) and its
  PBC behavior — the foundation everything else rests on.
- **ac-003 / ac-004** validate native-`Topology` network assembly and enforce the
  hard constraint that petgraph stays out of `compute` (it is `smiles`-only).
- **ac-005** validates the lifetime TCFs reused from `persist::pair_survival_tcf`.
- **ac-006** covers empty/degenerate inputs, double-counting, and the build gate.
