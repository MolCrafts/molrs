---
slug: chem-perceive-10-parmchk-tables
created: 2026-07-12
criteria:
  - id: ac-001
    summary: gaff_equiv and gaff_empirical are committed .rs
    type: code
    pass_when: |
      Both are generated as typed Rust consts preserving their semantics (equivalence/corr
              rows, weights, defaults; bond_power_m, bond_lnk, angle_zc). The generator byte-reproduces
              both.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: The last runtime text parse in the FF path is gone
    type: code
    pass_when: |
      `grep -rn 'serde_json' molrs/src/ff/` returns 0 hits. `grep -rn 'include_str!'
              molrs/src/ff/typifier/estimate/tables.rs` returns 0 hits. The two `.expect()` calls on
              embedded data are gone with them.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: The parmchk2 frcmod oracle is committed and RED
    type: code
    pass_when: |
      `molrs/tests/ff/typifier/parmchk2_oracle.rs` carries, for all 37 molecules, every term
              parmchk2 emits into its frcmod (type, tier, value). The corresponding test currently
              FAILS (it turns green in spec 11). The oracle regenerates from the committed script.
    status: pending
    last_checked: 
    evidence: 
---
