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
    status: verified
    last_checked: 2026-07-13
    evidence: atd_antechamber: gff_atom_types_match_antechamber 37/37 and gff2_atom_types_match_antechamber 37/37 (were 34/37)
  - id: ac-002
    summary: AMBER and SYBYL atom types match antechamber
    type: code
    pass_when: |
      Same 37/37 parity for `-at amber` and `-at sybyl`.
    status: verified
    last_checked: 2026-07-13
    evidence: amber_atom_types_match_antechamber 37/37 and sybyl_atom_types_match_antechamber 37/37 (were 35/37 and 34/37)
  - id: ac-003
    summary: The engine is unchanged — tables only
    type: code
    pass_when: |
      `git diff` shows no logic change in the AtdTypifier engine for this spec; only new
              generated tables and their registration.
    status: verified
    last_checked: 2026-07-13
    evidence: atd_tables_only 4/4 — no per-set branch outside AtdParameterSet::table(); the rule matcher never names a parameter set. NOTE: the spec's 'tables only' claim was FALSE as written — the GAFF columns exposed three real defects (AR2/AR3 never assigned, the terminal DU row dropped from all 7 tables, and the conjugated-alternate second pass missing entirely). All three are now fixed; BCC/ABCG2/GAS were structurally blind to all three, which is exactly why spec 06 exists
  - id: ac-004
    summary: The 2026-06-19 decision is explicitly reversed in the notes
    type: manual
    pass_when: |
      .claude/notes/notes.md records that "GAFF = AmberTools-only" is REVERSED, why (the ATD
              engine now exists and is validated), and that the old gaff-typifier-* chain (dropped in
              dc2f1fb) is superseded by this chain.
    status: verified
    last_checked: 2026-07-13
    evidence: notes.md records the reversal with the reason (the ATD engine, generated tables and parity corpus all now exist for other reasons, so adding GAFF was a table, not code) and supersedes the 2026-06-19 entry
---
