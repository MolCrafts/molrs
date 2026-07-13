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
    status: verified
    last_checked: 2026-07-13
    evidence: gaff.rs (7191 rows) + gaff2.rs (13098 rows) committed as typed static tables covering MASS/BOND/ANGLE/DIHE/IMPROPER/NONBON. 615 WILDCARD rows preserved per table (607 DIHE + 8 IMPROPER), emitted as Option<ParmType> = None so a wildcard cannot be misread as a concrete type. Generator byte-reproduces all 14 files; manifest hashes match
  - id: ac-002
    summary: No runtime text parsing of GAFF parameters
    type: code
    pass_when: |
      `grep -rn 'gaff.dat\|gaff2.dat' molrs/src` shows no include_str!/read/parse at runtime.
              The tables are `use`d directly. No .dat file ships in the repo.
    status: verified
    last_checked: 2026-07-13
    evidence: no runtime text parsing — no include_str!/read/parse of gaff.dat|gaff2.dat in molrs/src; no .dat ships in the repo. Provenance hashed into MANIFEST.sha256
  - id: ac-003
    summary: ForceField populates from GAFF/GAFF2 atom types
    type: code
    pass_when: |
      For the 37-molecule oracle typed with GAFF, every term with an exact match in gaff.dat is
              found and populated. A missing term returns Err (no silent fallback) — fallback lands in 11.
    status: verified
    last_checked: 2026-07-13
    evidence: every bond, angle, mass and NONBON term of all 37 oracle molecules exact-matches (zero misses, both tables). Misses are dihedrals covered only by wildcard rows (153 under gaff, 139 under gaff2) and return Err(GaffError::Missing) listing EVERY uncovered term, not the first — no silent fallback. End-to-end verified: typify -> gaff_forcefield -> to_frame -> to_potentials -> calc_energy_forces
  - id: ac-004
    summary: Binary and compile-time cost stay within the measured budget
    type: performance
    pass_when: |
      Adding the gaff + gaff2 tables costs no more than ~1.2 MB of stripped release binary and
              no more than ~1 s of clean build time (measured baseline: 15,474 rows = 1071 KB / 0.37 s).
    status: verified
    last_checked: 2026-07-13
    evidence: MEASURED on an isolated stripped release binary that consumes every field: +516.0 KB binary, +0.39 s clean build (budget 1.2 MB / 1 s). The naive &'static str layout BLEW the budget at +1415 KB (~60,500 type slots x a 16-byte fat pointer + load-time relocations); interning each slot to its MASS-row index (1 byte) is 2.7x smaller. Readability preserved — the generator emits a named const per type, so rows read ParmBondRow { i: T_C3, j: T_OH, .. }, never a bare number
---
