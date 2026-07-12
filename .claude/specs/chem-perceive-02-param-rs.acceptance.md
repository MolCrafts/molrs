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
    status: verified
    last_checked: 2026-07-12
    evidence: params::generator_byte_reproduces_the_committed_tables — re-runs the generator with $AMBERHOME set and asserts byte-identity across all 11 files; skips cleanly when unset (verified both ways)
  - id: ac-002
    summary: No runtime text parsing of parameter tables remains
    type: code
    pass_when: |
      `grep -rn 'include_str!' molrs/src/core/data.rs` returns no antechamber table.
              `grep -rn 'parse_str' molrs/src/ff/typifier/am1bcc.rs` returns 0 hits.
              The BCC correction table and ATD rules are `use`d as typed Rust consts.
    status: verified
    last_checked: 2026-07-12
    evidence: grep gates: parse_str in am1bcc.rs = 0; parse_environment/parse_atom_pattern/parse_pattern_list = 0; ANTECHAMBER_* include_str! in core/data.rs = 0. The env/property mini-language is pre-parsed into a static AST, so nothing is parsed at match time either
  - id: ac-003
    summary: Generated tables preserve GASPARM column semantics
    type: code
    pass_when: |
      The generated Gasteiger table names `a`, `b`, `c`, `chi_plus` (the `d` column) and
              `seed_charge` (the `formal_charge` column) as distinct typed fields — not an
              anonymous [f64; 5]. A test asserts chi_plus for H == 20.02 and for c3 == 19.04.
    status: verified
    last_checked: 2026-07-12
    evidence: params::gasteiger_columns_keep_their_semantics — chi_plus(H) == 20.02 (NOT a+b+c = 12.85), chi_plus(c3) == 19.04 == a+b+c; a/b/c, chi_plus and seed_charge are three distinct named fields
  - id: ac-004
    summary: The .rs is the single in-repo source of truth — no .DAT intermediate
    type: code
    pass_when: |
      molrs/data/antechamber/ no longer exists. The generator reads its source tables from
              $AMBERHOME. The drift guard runs only when $AMBERHOME is set and skips cleanly otherwise.
    status: verified
    last_checked: 2026-07-12
    evidence: molrs/data/antechamber/ deleted; the 11 committed .rs under ff/params/generated/ are the single in-repo source of truth; the generator reads $AMBERHOME
  - id: ac-005
    summary: Binary and compile-time cost stay within the measured budget
    type: performance
    pass_when: |
      The generated tables add no more than ~1.2 MB to a stripped release binary and no more
              than ~1 s to a clean build. (Measured baseline: 15,474 rows of gaff+gaff2+BCCPARM cost
              1071 KB and 0.37 s; molrs already embeds 3974 KB of raw include_str! text today.)
    status: verified
    last_checked: 2026-07-12
    evidence: measured on an isolated stripped release binary that consumes the tables: +234.7 KB (budget <=1.2 MB) and +0.21 s clean build (budget <=1 s)
---
