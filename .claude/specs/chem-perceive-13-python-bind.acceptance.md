---
slug: chem-perceive-13-python-bind
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Native AM1-BCC is reachable from Python
    type: code
    pass_when: |
      `molrs.BccModel(parameter_set="bcc").correct(mol, am1)` returns an ndarray of charges.
              On the 37-molecule oracle the Python result is bitwise identical to the Rust result.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: Perceive and AtdTypifier are exposed
    type: code
    pass_when: |
      `molrs.Perceive().find_rings(g)` returns a graph. `molrs.AtdTypifier(parameter_set="gaff")`
              types a molecule and matches `antechamber -at gff`.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: All three charge models are bound
    type: code
    pass_when: |
      BccModel, MullikenModel and GasteigerModel are all importable from `molrs` and all
              implement the same Python-side calling convention.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: The compat alias is gone
    type: code
    pass_when: |
      `grep -rn 'molrs::chem' .` returns 0 hits across molrs, molrs-python, molrs-cxxapi,
              molrs-ffi and molrs-wasm. `pub use crate::perceive as chem;` is removed from
              molrs/src/lib.rs. Existing Python tests are green with only import paths changed.
    status: pending
    last_checked: 
    evidence: 
---
