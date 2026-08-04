---
spec: release-0-12-06-docs-surface
created: 2026-08-04
criteria:
  - id: ac-001
    summary: no parse_smiles free-fn in site-src/README
    type: docs
    pass_when: "grep finds no molrs.parse_smiles in molrs-python README/site-src; SmilesIR used instead"
    status: verified
    last_checked: 2026-08-04
  - id: ac-002
    summary: force-field guide uses calc_energy_forces
    type: docs
    pass_when: "guides teach calc_energy_forces / typify compose; no potentials.eval or OPLS build"
    status: verified
    last_checked: 2026-08-04
  - id: ac-003
    summary: single-crate rust reference
    type: docs
    pass_when: "reference/rust.md lists only molcrafts-molrs (+ optional cxxapi note)"
    status: verified
    last_checked: 2026-08-04
  - id: ac-004
    summary: version pins are 0.12
    type: docs
    pass_when: "README/interop/lib.rs/quickstart-rust pins use 0.12 not 0.0.15/0.10/0.11"
    status: verified
    last_checked: 2026-08-04
  - id: ac-005
    summary: transport docs match science contracts
    type: docs
    pass_when: "transport guide states unbiased VACF, D=1/d∫C, MSD unwrapped, analysis time fs"
    status: verified
    last_checked: 2026-08-04
out_of_scope:
  - molpy documentation
---

# Acceptance — release-0-12-06-docs-surface

Published molrs docs and examples teach only the real 0.12 API and science contracts.
