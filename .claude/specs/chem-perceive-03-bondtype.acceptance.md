---
slug: chem-perceive-03-bondtype
created: 2026-07-12
criteria:
  - id: ac-001
    summary: BCC bond types match antechamber on the whole oracle
    type: code
    pass_when: |
      `bcc_bond_types_match_antechamber` passes 323/323 bonds across all 37 molecules.
              (Sanctioned fallback, ONLY if the literal Kekulé order proves brittle AND ac-002 is
              green: relax the assertion to treat {7,8,10} as one equivalence class.)
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: BCC increments are Kekule-invariant (licenses the fallback)
    type: code
    pass_when: |
      For every (i,j) atom-type pair present in BCCPARM: bcc(i,j,7) == bcc(i,j,8) ==
              bcc(i,j,10) exactly (f64 equality), over the whole 405-row table.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: Delocalized (type 9) perception fixes the carboxylate/nitro symmetry break
    type: code
    pass_when: |
      acetate: both C–O bonds get type 9. nitromethane: both N–O bonds get type 9.
              Consequently the two acetate oxygens receive IDENTICAL charges downstream.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Aromatic promotion boundary is correct
    type: code
    pass_when: |
      Biphenyl's inter-ring bond is NOT promoted (stays 1 or 2). A 7-membered aromatic ring's
              bonds are NOT promoted. Benzene/pyridine/imidazole/thiophene ring bonds ARE in {7,8}.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-005
    summary: Types 6 and 11 are investigated, not guessed
    type: code
    pass_when: |
      The spec records whether types 6 and 11 are reachable in AmberTools25. Any unreachable
              type is asserted unreachable by a test; no dead branch is implemented for it.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-006
    summary: Clean-room / licensing posture is documented
    type: manual
    pass_when: |
      A written decision (in .claude/notes/) records the licensing posture for reimplementing
              antechamber's GPL bondtype.c, reviewed before merge.
    status: verified
    last_checked: 2026-07-12
    evidence: owner decision recorded in .claude/notes/notes.md (2026-07-12) — proceed on an educational/research basis with the AmberTools developers' permission to read their source; the BSD-3 vs GPL-3 context and the source-derived (not clean-room) nature of the work are recorded there. Explicitly waived by the owner before spec 03 began.
---
