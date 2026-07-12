---
slug: chem-perceive-01-layer
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Existing suite is green with zero behaviour delta
    type: code
    pass_when: |
      `cargo test --all-features` passes. No existing assertion value in
              tests/core/{aromaticity,rings,hydrogens}.rs or tests/embed/* is modified —
              `git diff` on those files shows only import-path changes.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: Perceive builder is graph-in / graph-out
    type: code
    pass_when: |
      `Perceive::new().find_rings(&g)` returns a `MolGraph` carrying the ring component;
              the input `g` is not mutated. Same shape for find_aromaticity / find_hydrogens /
              find_stereo / find_rotatable.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: perceive is always compiled, not feature-gated
    type: code
    pass_when: |
      `cargo build --no-default-features` compiles `molrs::perceive`. CLAUDE.md's module
              table documents the deviation from the "core always on, rest gated" rule.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Binder crates still build via the compat alias
    type: code
    pass_when: |
      `cargo check` in the separate molrs-python workspace succeeds with no source change:
              `molrs::chem::aromaticity::perceive_aromaticity` still resolves.
    status: pending
    last_checked: 
    evidence: 
---
