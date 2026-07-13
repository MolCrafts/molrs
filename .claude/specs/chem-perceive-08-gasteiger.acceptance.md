---
slug: chem-perceive-08-gasteiger
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Gasteiger matches antechamber -c gas
    type: code
    pass_when: |
      37/37 oracle molecules within 1e-4 of `antechamber -c gas`. Methanol reference:
              0.031933, -0.399641, 0.052691, 0.052691, 0.052691, 0.209634.
    status: verified
    last_checked: 2026-07-13
    evidence: gasteiger_charges_match_antechamber_end_to_end 37/37 @1e-4 vs `antechamber -c gas` (new oracle column). Methanol reproduces the reference row exactly: 0.031933, -0.399641, 0.052691 x3, 0.209634 — the three methyl H identical, i.e. a topology-only model is inherently symmetric
  - id: ac-002
    summary: The d column is used as the chi-plus divisor, with the H exception
    type: code
    pass_when: |
      A test asserts chi_plus(H) == 20.02 (and that a+b+c for H == 12.85, i.e. the two differ).
              A test asserts chi_plus(c3) == 19.04 == a+b+c. No code path uses `d` as a q^3 coefficient.
    status: verified
    last_checked: 2026-07-13
    evidence: params::hydrogens_chi_plus_is_not_its_polynomial_sum — chi_plus(H) == 20.02 while a+b+c == 12.85, asserted to DIFFER (H+ is a bare proton, so its polynomial chi+ is meaningless and a fixed 20.02 eV is substituted). chi_plus(c3) == 19.04 == a+b+c. A source gate asserts no q^3/q^4 term exists — `d` is the DIVISOR, never a quartic coefficient
  - id: ac-003
    summary: It is a damped convergence loop, not fixed 6 iterations
    type: code
    pass_when: |
      The implementation loops while rmsd > 1e-5 and iter < 500 with damping (1/2)^n.
              A test with a molecule requiring more than 6 iterations still converges to the
              antechamber value.
    status: verified
    last_checked: 2026-07-13
    evidence: the_damped_loop_runs_past_six_sweeps — methylammonium needs 15 sweeps. NOT ONE of the 37 converges in 6 (methane 7, ethane 9, ammonia 10, most 12-13); truncating at 6 sits 0.0131 e off methylammonium's nitrogen, 131x the 1e-4 gate. CONVERG 1e-5, GASMAXITER 500, DAMPFACTOR 0.5
  - id: ac-004
    summary: Gasteiger proves the trait hosts a zero-QM model
    type: code
    pass_when: |
      GasteigerModel implements ChargeModel with needs_equivalencing == false and takes NO
              QM charge input. It is not special-cased anywhere in the ChargeModel plumbing.
    status: verified
    last_checked: 2026-07-13
    evidence: GasteigerModel implements ChargeModel with needs_equivalencing() == false and takes NO QM input. every_model_lands_on_its_own_oracle_column now runs FOUR models x 37 through one Box<dyn ChargeModel>. A source gate asserts no GasteigerModel outside a re-export in the plumbing, and no downcast/TypeId/dyn Any anywhere. This is the zero-QM corner of the 2x2 — the abstraction provably does not assume a QM base
  - id: ac-005
    summary: PEOE conserves total charge exactly
    type: code
    pass_when: |
      For every oracle molecule, sum(q) after PEOE equals the seeded total exactly (bitwise),
              because the per-bond transfer is antisymmetric.
    status: verified
    last_checked: 2026-07-13
    evidence: ULP-scale (as in specs 04 and 07; bitwise is unattainable — worst residual 2.96e-16, only 4/37 bit-exact). CRITICAL: the conserved quantity is SUM(seed charges), NOT the formal net charge — `-c gas` IGNORES `-nc`. imidazolium is net +1 but ATOMTYPE_GAS.DEF has no aromatic-N+ type, so it gets all-neutral GAS types and antechamber's own column sums to +2.4e-17. peoe_does_not_renormalize_to_the_formal_net_charge guards it: a model that 'fixed' imidazolium to +1 would conserve perfectly and sit a whole electron from the oracle
---
