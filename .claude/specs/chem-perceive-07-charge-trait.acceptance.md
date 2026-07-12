---
slug: chem-perceive-07-charge-trait
created: 2026-07-12
criteria:
  - id: ac-001
    summary: End-to-end AM1-BCC charges match antechamber
    type: code
    pass_when: |
      `am1bcc_charges_match_antechamber_end_to_end` passes 37/37 within 1e-4 (currently 4/37
              fail: acetate max|dq|=0.2014, nitromethane/pyridine/imidazole hard error).
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: ABCG2 and Mulliken go through the same trait
    type: code
    pass_when: |
      ABCG2 end-to-end matches `antechamber -c abcg2` 37/37 @1e-4. MullikenModel returns the
              supplied QM charges unchanged (exact equality with the oracle's am1_charges). Neither is
              special-cased in the trait.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: The pull-trait backend seam is deleted
    type: code
    pass_when: |
      `grep -rnE 'AM1ChargeBackend|UnavailableAM1Backend|normalize_total_charge' molrs/src`
              returns 0 hits. The public API is `BccModel::correct(&mol, &am1) -> Result<Vec<f64>>`:
              a pure function taking a slice, with no backend trait and no mutation of `mol`.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Charge assignment never clobbers force-field atom types
    type: code
    pass_when: |
      A molecule carrying GAFF atom types (e.g. "c3","oh","h1") in keys::TYPE has the SAME
              keys::TYPE values after BccModel::correct. Bond types likewise. BCC codes never appear
              in keys::TYPE.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-005
    summary: BCC conserves total charge exactly and never renormalizes
    type: code
    pass_when: |
      For every oracle molecule, sum(q_final) - sum(q_after_equivalencing) == 0 bitwise.
              |sum(q_final) - net_charge| <= 0.005 is asserted as a TOLERANCE; no code path shifts,
              rescales or rounds charges to reach the integer net charge.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-006
    summary: The correction engine is unchanged and still green
    type: code
    pass_when: |
      `bcc_corrections_match_antechamber_given_reference_types` remains 37/37 green — the
              BCCPARM lookup and the smaller-BCC-atom-type sign convention were not modified.
    status: pending
    last_checked: 
    evidence: 
---
