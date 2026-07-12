---
slug: chem-perceive-14-all-tables
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Every parameter table is a committed .rs
    type: code
    pass_when: |
      mmff94, mmff94s, oplsaa, rigid-fragments and ring-fragments are all generated typed Rust
              tables. `grep -rn 'include_str!' molrs/src` returns 0 hits. The molrs/data/ directory no
              longer exists.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: Pure representation change — zero numerical delta
    type: code
    pass_when: |
      The MMFF94/MMFF94s/OPLS typifier and potential test suites, and the conformer/ETKDG
              fragment tests, all pass with NO assertion value changed. `git diff` on those test files
              shows no numeric edits.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: Generator byte-reproduces every table
    type: code
    pass_when: |
      With $AMBERHOME (and any other upstream source) available, the generator regenerates all
              tables and `git diff --exit-code` is clean. Skips cleanly when sources are unavailable.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Final binary size and build time are measured and recorded
    type: performance
    pass_when: |
      The spec records the measured stripped-binary size and clean-build time before and after.
              The net binary delta is expected to be SMALL or negative (the raw-text copies and their
              runtime parsers are removed); any regression beyond +2 MB total must be justified.
    status: pending
    last_checked: 
    evidence: 
---
