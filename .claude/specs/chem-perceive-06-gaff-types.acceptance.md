---
slug: chem-perceive-06-gaff-types
created: 2026-07-12
criteria:
  - id: ac-001
    summary: GAFF and GAFF2 atom types match antechamber
    type: code
    pass_when: |
      For all 37 oracle molecules, `AtdTypifier` with the GFF table reproduces
              `antechamber -at gff` 37/37, and with GFF2 reproduces `antechamber -at gff2` 37/37.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: AMBER and SYBYL atom types match antechamber
    type: code
    pass_when: |
      Same 37/37 parity for `-at amber` and `-at sybyl`.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: The engine is unchanged — tables only
    type: code
    pass_when: |
      `git diff` shows no logic change in the AtdTypifier engine for this spec; only new
              generated tables and their registration.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: The 2026-06-19 decision is explicitly reversed in the notes
    type: manual
    pass_when: |
      .claude/notes/notes.md records that "GAFF = AmberTools-only" is REVERSED, why (the ATD
              engine now exists and is validated), and that the old gaff-typifier-* chain (dropped in
              dc2f1fb) is superseded by this chain.
    status: pending
    last_checked: 
    evidence: 
---
