---
slug: chem-perceive-12-cxx-bridge
created: 2026-07-12
criteria:
  - id: ac-001
    summary: The bridge returns Result and never aborts on user chemistry
    type: code
    pass_when: |
      The cxx bridge fn is declared `-> Result<Vec<f64>>`. A molecule whose chemistry has no
              BCC parameter (e.g. containing boron) raises a catchable `rust::Error` on the C++ side;
              the process does NOT abort. A test demonstrates the catch.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: ABCG2 is reachable from C++
    type: code
    pass_when: |
      The bridge takes a parameter-set selector; passing ABCG2 produces ABCG2 charges.
              `grep -rn 'abcg2\|parameter_set' molrs-cxxapi/src` returns hits (currently 0).
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: The fake backend and the normalize argument are gone
    type: code
    pass_when: |
      `grep -rnE 'ProvidedAM1ChargeBackend|normalize_total_charge' molrs-cxxapi/src` returns
              0 hits.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: The Atomiverse companion change is declared
    type: manual
    pass_when: |
      The spec names the exact Atomiverse edit required (am1_bcc_charge_assigner.cpp +
              AM1BCCChargeConfig) and it is tracked as a cross-repo dependency, not silently assumed.
    status: pending
    last_checked: 
    evidence: 
---
