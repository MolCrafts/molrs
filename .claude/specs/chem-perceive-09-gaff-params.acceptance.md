---
slug: chem-perceive-09-gaff-params
created: 2026-07-12
criteria:
  - id: ac-001
    summary: gaff.dat and gaff2.dat are committed .rs tables
    type: code
    pass_when: |
      Both are generated as typed Rust static tables covering MASS/BOND/ANGLE/DIHE/IMPROPER/
              NONBON, including wildcard (X) rows. The generator byte-reproduces both; git diff is clean.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: No runtime text parsing of GAFF parameters
    type: code
    pass_when: |
      `grep -rn 'gaff.dat\|gaff2.dat' molrs/src` shows no include_str!/read/parse at runtime.
              The tables are `use`d directly. No .dat file ships in the repo.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: ForceField populates from GAFF/GAFF2 atom types
    type: code
    pass_when: |
      For the 37-molecule oracle typed with GAFF, every term with an exact match in gaff.dat is
              found and populated. A missing term returns Err (no silent fallback) — fallback lands in 11.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Binary and compile-time cost stay within the measured budget
    type: performance
    pass_when: |
      Adding the gaff + gaff2 tables costs no more than ~1.2 MB of stripped release binary and
              no more than ~1 s of clean build time (measured baseline: 15,474 rows = 1071 KB / 0.37 s).
    status: pending
    last_checked: 
    evidence: 
---
