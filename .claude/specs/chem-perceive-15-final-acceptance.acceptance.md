---
slug: chem-perceive-15-final-acceptance
created: 2026-07-14
closed: 2026-08-04
criteria:
  - id: ac-001
    summary: ONE place, ONE form — and the word `generated` is gone from every identifier
    type: code
    pass_when: "params flat; no include_str!; no generated/generator identifiers"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      cargo test -p molcrafts-molrs --features full --test architecture_gate (ac001_*)
  - id: ac-002
    summary: Nothing parses parameter text at runtime
    type: code
    pass_when: "no runtime table parse on force-field path"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      architecture_gate ac002_*
  - id: ac-003
    summary: ONE perception layer, ONE interpolation seam, ONE MMFF path
    type: code
    pass_when: "no chem alias; one ParameterInterpolator; no build_mmff_potentials"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      architecture_gate ac003_*
  - id: ac-004
    summary: A registered kernel constructor that ignores tp is not a Style
    type: runtime
    pass_when: "ParamSource PerInstance bidirectional semantic gate"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      architecture_gate ac004
  - id: ac-005
    summary: Chain against real external oracles (offline goldens)
    type: runtime
    pass_when: "hardcoded goldens from AmberTools/RDKit offline dumps; no live tools in CI"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      molpy tests/test_typifier/test_atd.py (6 ATD tables + Gasteiger float64)
      molpy tests/test_typifier/test_mmff.py (RDKit-locked ethane types/energy, ethanol multiset)
      cxxapi tests/am1bcc_reference.rs + am1bcc_bridge (full charge matrix fixture)
  - id: ac-006
    summary: Python reproduces Rust BIT-FOR-BIT (float64)
    type: runtime
    pass_when: "Python bindings return float64 and exact type strings / charge atol 1e-6 fixture / sum 1e-12"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      test_atd.py gasteiger float64 + atol 1e-6 vs reference; charge sum < 1e-12
      test_atd.py type lists exact; test_mmff.py ethane types exact list equality
  - id: ac-007
    summary: REVERSE gates
    type: runtime
    pass_when: "zero elec; MMFF94==94s ethane; benzene impropers; equal O charges"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      architecture_gate reverse module
  - id: ac-008
    summary: No subset assertion without stated reason
    type: code
    pass_when: "typifier tests declare # subset reason: when cherry-picking"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      molpy/tests/test_typifier/test_fixture_subset_discipline.py
  - id: ac-009
    summary: Every gate has been PROVEN to bite
    type: runtime
    pass_when: "gates use non-self-exempting needles"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      architecture_gate concat! needles; subset discipline gate
  - id: ac-010
    summary: Acceptance fixed NOTHING in production for this close
    type: code
    pass_when: "gates/tests only for this acceptance close"
    status: verified
    last_checked: "2026-08-04"
    verified_by: agent-auto
    evidence: |
      Final close adds only molpy tests + acceptance metadata
out_of_scope:
  - Live AmberTools/RDKit in CI
---

# Acceptance — chem-perceive-15-final-acceptance

All criteria verified for 0.12 close. Goldens are offline fixtures; live tools
are not part of the test suite.
