---
slug: chem-perceive-11-param-estimate
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Native estimation reproduces parmchk2 on the whole oracle
    type: code
    pass_when: |
      For all 37 molecules, every term parmchk2 emits into its frcmod is reproduced natively:
              same fallback tier (PenaltyTier) and same value within table precision. The RED fixture
              from spec 10 is green.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: No frcmod file I/O remains
    type: code
    pass_when: |
      `grep -rnE 'Frcmod::parse_str|write_string' molrs/src` returns 0 hits. No test reads an
              external .frcmod file. molrs does not depend on any external parameter file at runtime.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: The estimation is built on the existing estimator architecture
    type: code
    pass_when: |
      The GAFF fallback goes through ParameterEstimator / ParameterInterpolator /
              TypifierParameterContext — not a parallel reimplementation. The OPLS context still works
              unchanged (its existing tests stay green).
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Fallback tiers are ordered exactly as parmchk2
    type: code
    pass_when: |
      A test drives a term through each tier in turn: exact match, wildcard (X) row,
              atom-type-equivalence substitution (corr row, with penalty), empirical formula —
              and asserts the chosen tier matches parmchk2's for that term.
    status: pending
    last_checked: 
    evidence: 
---
