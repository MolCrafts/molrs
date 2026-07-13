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
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `molrs/src/ff/params/generated/{gaff_equiv,gaff_empirical}.rs` committed, emitted by
      `scripts/gen_param_tables.py` from PARMCHK.DAT + PARM_BLBA_GAFF{,2}.DAT (both added to the
      hashed source set). `params::{gaff_equiv_and_empirical_are_generated_tables,
      committed_tables_match_the_manifest_hashes}` green; the generator diffs clean on regeneration
      with $AMBERHOME set (16/16 files byte-identical). `molrs/data/gaff_{equiv,empirical}.json` deleted.
  - id: ac-002
    summary: The last runtime text parse in the FF path is gone
    type: code
    pass_when: |
      No runtime text parse remains in the FF path: `grep -rn 'include_str!'
              molrs/src/ff/typifier/estimate/tables.rs` returns 0 hits, no `serde_json` *deserializer*
              entry point (`from_str` / `from_slice` / `from_reader`) is reachable from `molrs/src/ff/`,
              and the two `.expect()` calls on embedded data are gone with them.
              AMENDED 2026-07-14: originally "`grep -rn 'serde_json' molrs/src/ff/` returns 0 hits".
              That substring grep was a bad proxy for this criterion's own summary — it also catches
              `forcefield_meta.rs`, which only *constructs* JSON (`json!`) to emit provenance metadata
              and never parses anything. The binding tests
              `estimate_tables_generated::{the_force_field_tree_never_deserializes_json_at_runtime,
              the_estimate_tables_embed_no_table_text, the_estimate_tables_never_expect_on_embedded_data}`
              enforce the parse ban directly, which is stricter than the substring grep.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      tables.rs rewritten to `use` the committed `params::generated::{gaff_equiv, gaff_empirical}`;
      zero `include_str!`, zero deserializer call, zero `.expect()` on embedded data in production code.
      The three `estimate_tables_generated` tests are green.
  - id: ac-003
    summary: The parmchk2 frcmod oracle is committed and RED
    type: code
    pass_when: |
      `molrs/tests/ff/typifier/parmchk2_oracle.rs` carries, for all 37 molecules, every term
              parmchk2 emits into its frcmod (type, tier, value). The corresponding test currently
              FAILS (it turns green in spec 11). The oracle regenerates from the committed script.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `parmchk_terms` column added to the oracle by `scripts/gen_am1bcc_oracle.py` (real parmchk2 run,
      gaff + gaff2, all 37 molecules); `molrs/tests/ff/typifier/parmchk2_oracle.rs` (596 lines) consumes it.
      It was RED at spec-10 close (4 failures) and is green after spec 11.
---
