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
    status: verified
    last_checked: 2026-07-12
    evidence: bcc_bond_types_match_antechamber — 323/323 bonds, 37/37 molecules (was 11/37 disagreeing). Kekule phase reproduced via penalty-minimising kekulization over the aromatic subsystem using antechamber's own APS numbers
  - id: ac-002
    summary: BCC increments are Kekule-invariant (licenses the fallback)
    type: code
    pass_when: |
      For every (i,j) atom-type pair present in BCCPARM: bcc(i,j,7) == bcc(i,j,8) ==
              bcc(i,j,10) exactly (f64 equality), over the whole 405-row table.
    status: verified
    last_checked: 2026-07-12
    evidence: typifier::bcc_bond_type::bcc_corrections_are_identical_for_bond_types_7_8_and_10 — exact f64 equality over the whole 405-row BCCPARM table
  - id: ac-003
    summary: Delocalized (type 9) perception fixes the carboxylate/nitro symmetry break
    type: code
    pass_when: |
      acetate: both C–O bonds get type 9. nitromethane: both N–O bonds get type 9.
              Consequently the two acetate oxygens receive IDENTICAL charges downstream.
    status: verified
    last_checked: 2026-07-12
    evidence: typifier::bcc_bond_type::acetate_carboxylate_oxygens_receive_identical_charges (the 0.2014 e break is gone); both acetate C-O and both nitromethane N-O are type 9
  - id: ac-004
    summary: Aromatic promotion boundary is correct
    type: code
    pass_when: |
      Biphenyl's inter-ring bond is NOT promoted (stays 1 or 2). A 7-membered aromatic ring's
              bonds are NOT promoted. Benzene/pyridine/imidazole/thiophene ring bonds ARE in {7,8}.
    status: verified
    last_checked: 2026-07-12
    evidence: perceive::bond_type — biphenyl inter-ring bond and 7-membered aromatic ring bonds stay 1/2; benzene/pyridine/imidazole/thiophene ring bonds are in {7,8}
  - id: ac-005
    summary: Types 6 and 11 are investigated, not guessed
    type: code
    pass_when: |
      The spec records whether types 6 and 11 are reachable in AmberTools25. Any unreachable
              type is asserted unreachable by a test; no dead branch is implemented for it.
    status: verified
    last_checked: 2026-07-12
    evidence: Measured against AmberTools25 over 33 probes: emitted types are {1,2,3,6,7,8,9}, never 10 or 11. Type 11 asserted unreachable (no dead branch implemented); type 6 reachable (nitrate/nitrite/pyridine-N-oxide) and implemented
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
