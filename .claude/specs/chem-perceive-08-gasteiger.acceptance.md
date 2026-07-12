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
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: The d column is used as the chi-plus divisor, with the H exception
    type: code
    pass_when: |
      A test asserts chi_plus(H) == 20.02 (and that a+b+c for H == 12.85, i.e. the two differ).
              A test asserts chi_plus(c3) == 19.04 == a+b+c. No code path uses `d` as a q^3 coefficient.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: It is a damped convergence loop, not fixed 6 iterations
    type: code
    pass_when: |
      The implementation loops while rmsd > 1e-5 and iter < 500 with damping (1/2)^n.
              A test with a molecule requiring more than 6 iterations still converges to the
              antechamber value.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Gasteiger proves the trait hosts a zero-QM model
    type: code
    pass_when: |
      GasteigerModel implements ChargeModel with needs_equivalencing == false and takes NO
              QM charge input. It is not special-cased anywhere in the ChargeModel plumbing.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-005
    summary: PEOE conserves total charge exactly
    type: code
    pass_when: |
      For every oracle molecule, sum(q) after PEOE equals the seeded total exactly (bitwise),
              because the per-bond transfer is antisymmetric.
    status: pending
    last_checked: 
    evidence: 
---
