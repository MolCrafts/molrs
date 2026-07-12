---
slug: chem-perceive-04-equivalence
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Equivalencing reproduces antechamber from raw Mulliken
    type: code
    pass_when: |
      Averaging the oracle's `am1_charges_raw` (raw sqm Mulliken) by the computed classes
              reproduces the oracle's `am1_charges` for all 37 molecules within 1e-4. The 20
              molecules that antechamber actually changes are all changed identically.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: Classes come from path-scores, not automorphism orbits
    type: code
    pass_when: |
      The implementation enumerates simple paths to terminal atoms and compares sorted score
              arrays. A test constructs a molecule where automorphism orbits are STRICTLY FINER than
              antechamber's path-score classes and asserts molrs merges them the way antechamber does.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: Score comparison is exact, not tolerance-based
    type: code
    pass_when: |
      A dedicated test constructs two atoms whose sorted score arrays differ by less than any
              plausible tolerance and asserts they are NOT merged.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Averaging conserves total charge exactly
    type: code
    pass_when: |
      For every oracle molecule, sum(charges) before and after equivalencing are bitwise equal.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-005
    summary: Methanol methyl hydrogens become identical
    type: code
    pass_when: |
      Methanol: the three methyl H charges after equivalencing are all exactly equal
              (0.068 each from raw 0.053/0.098/0.053).
    status: pending
    last_checked: 
    evidence: 
  - id: ac-006
    summary: Clean-room / licensing posture is documented
    type: manual
    pass_when: |
      A written decision records the licensing posture for reimplementing antechamber's
              GPL equatom.c, reviewed before merge.
    status: pending
    last_checked: 
    evidence: 
---
