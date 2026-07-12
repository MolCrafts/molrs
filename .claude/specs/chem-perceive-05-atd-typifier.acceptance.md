---
slug: chem-perceive-05-atd-typifier
created: 2026-07-12
criteria:
  - id: ac-001
    summary: BCC atom types match antechamber on the whole oracle
    type: code
    pass_when: |
      `bcc_atom_types_match_antechamber` passes 37/37 molecules (currently 2/37 fail:
              pyridine and imidazole aromatic N typed "25" instead of "24").
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: One engine drives multiple type tables
    type: code
    pass_when: |
      `AtdTypifier` is constructed with a parameter-set selector. BCC, ABCG2 and GAS atom
              types each reproduce `antechamber -at {bcc,abcg2,gas}` 37/37, with no per-table
              special-casing in the engine.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: No runtime table parsing on the typify path
    type: code
    pass_when: |
      `grep -rn 'parse_str' molrs/src/ff/typifier/` returns 0 hits. The ATD rules come from
              the generated `.rs` tables. Typifying 500 molecules does not re-parse anything.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: The empty-table footgun is gone
    type: code
    pass_when: |
      There is no constructor that yields a typifier/corrector with an empty parameter table.
              Constructing a model without a parameter set is a compile error or returns Err.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-005
    summary: Clean-room / licensing posture is documented
    type: manual
    pass_when: |
      A written decision records the licensing posture for reimplementing antechamber's
              GPL atomtype.c, reviewed before merge.
    status: pending
    last_checked: 
    evidence: 
---
