---
slug: chem-perceive-02-param-rs
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Generator byte-reproduces every committed .rs
    type: code
    pass_when: |
      Running `scripts/gen_param_tables.py` regenerates all 10 tables and `git diff --exit-code`
              on molrs/src/ff/params/generated/ reports no change.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: No runtime text parsing of parameter tables remains
    type: code
    pass_when: |
      `grep -rn 'include_str!' molrs/src/core/data.rs` returns no antechamber table.
              `grep -rn 'parse_str' molrs/src/ff/typifier/am1bcc.rs` returns 0 hits.
              The BCC correction table and ATD rules are `use`d as typed Rust consts.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: Generated tables preserve GASPARM column semantics
    type: code
    pass_when: |
      The generated Gasteiger table names `a`, `b`, `c`, `chi_plus` (the `d` column) and
              `seed_charge` (the `formal_charge` column) as distinct typed fields — not an
              anonymous [f64; 5]. A test asserts chi_plus for H == 20.02 and for c3 == 19.04.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: The .rs is the single in-repo source of truth — no .DAT intermediate
    type: code
    pass_when: |
      molrs/data/antechamber/ no longer exists. The generator reads its source tables from
              $AMBERHOME. The drift guard runs only when $AMBERHOME is set and skips cleanly otherwise.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-005
    summary: Binary and compile-time cost stay within the measured budget
    type: performance
    pass_when: |
      The generated tables add no more than ~1.2 MB to a stripped release binary and no more
              than ~1 s to a clean build. (Measured baseline: 15,474 rows of gaff+gaff2+BCCPARM cost
              1071 KB and 0.37 s; molrs already embeds 3974 KB of raw include_str! text today.)
    status: pending
    last_checked: 
    evidence: 
---
