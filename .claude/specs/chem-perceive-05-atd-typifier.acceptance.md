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
    status: verified
    last_checked: 2026-07-13
    evidence: bcc_atom_types_match_antechamber 37/37 (unchanged); the_bcc_typifier_agrees_atom_for_atom_with_the_atd_engine proves BCCAtomTypifier IS AtdTypifier with the BCC table, not a second engine
  - id: ac-002
    summary: One engine drives multiple type tables
    type: code
    pass_when: |
      `AtdTypifier` is constructed with a parameter-set selector. BCC, ABCG2 and GAS atom
              types each reproduce `antechamber -at {bcc,abcg2,gas}` 37/37, with no per-table
              special-casing in the engine.
    status: verified
    last_checked: 2026-07-13
    evidence: atd_antechamber: abcg2_atom_types_match_antechamber 37/37 and gas_atom_types_match_antechamber 37/37, both through the generic engine with no per-set special-casing. GAS has an ATD table but NO BCC correction table, which is why AtdParameterSet (7 variants) had to be split out of BccParameterSet (Bcc|Abcg2 only)
  - id: ac-003
    summary: No runtime table parsing on the typify path
    type: code
    pass_when: |
      `grep -rn 'parse_str' molrs/src/ff/typifier/` returns 0 hits. The ATD rules come from
              the generated `.rs` tables. Typifying 500 molecules does not re-parse anything.
    status: verified
    last_checked: 2026-07-13
    evidence: atd_no_runtime_parse: parse_str grep gate = 0 hits under src/ff/typifier/, no-runtime-file-read gate, and typify_cost_is_independent_of_table_size (counting global_allocator: typifying 500 molecules costs the same with a 46-rule table as with a 200-rule one; a re-parse would allocate per rule)
  - id: ac-004
    summary: The empty-table footgun is gone
    type: code
    pass_when: |
      There is no constructor that yields a typifier/corrector with an empty parameter table.
              Constructing a model without a parameter set is a compile error or returns Err.
    status: verified
    last_checked: 2026-07-13
    evidence: parameter_set_required 4/4. All five footguns deleted: Default derive on BCCCorrectionTable, BCCCorrectionTable::new(), impl Default for BCCCorrector, impl Default for AM1BCCTypifier, and the 1-arg AM1BCCTypifier::new(). Replaced by BCCCorrectionTable::from_rows(&[BccCorrectionRow]) and AM1BCCTypifier::new(backend, table) — both NAME their content. every_surviving_constructor_builds_a_populated_table asserts it behaviourally, not just by source scan
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
