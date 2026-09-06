---
slug: amber-prmtop-complete-03-lammps
criteria:
  - id: ac-001
    summary: LAMMPS writer always emits an explicit special_bonds line after units
    type: code
    pass_when: |
      In molrs/src/ff/forcefield/writers/lammps.rs, LammpsFfWriter::write_str output for a
      ForceField with special_bonds {lj:[0,0,0.5], coul:[0,0,5/6]} contains the line
      "special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333"
      positioned after the "units" line and before the first *_coeff line, and still
      contains it when LammpsWriteOptions.skip_units is true. Both cases are asserted by
      #[cfg(test)] tests in that file.
    status: pending
  - id: ac-002
    summary: No new writer option and no binder edit
    type: code
    pass_when: |
      LammpsWriteOptions in molrs/src/ff/forcefield/writers/lammps.rs declares exactly its
      ten current fields (precision, skip_pair_style, skip_units, units, atom_types,
      bond_types, angle_types, dihedral_types, improper_types, type_ids) — no
      skip_special_bonds — and the phase diff touches no file under molrs-python/src/.
    status: pending
  - id: ac-003
    summary: LAMMPS reader parses presets and explicit weight triples
    type: code
    pass_when: |
      #[cfg(test)] tests in molrs/src/ff/forcefield/readers/lammps.rs show read_str yields:
      amber -> lj [0,0,0.5] and |coul_14 - 5/6| < 1e-12; charmm -> both [0,0,0];
      dreiding -> both [0,0,1.0]; fene -> both [0,1.0,1.0]; "lj 0.0 0.0 0.5 coul 0.0 0.0
      0.8333333333333334" -> the same numbers; a bare triple applies to both kinds; a
      trailing "angle yes dihedral no" is accepted; an unrecognised token is an Err.
    status: pending
  - id: ac-004
    summary: Undeclared special_bonds is an Err and the AMBER constants are gone
    type: code
    pass_when: |
      LammpsFfReader::read_str on a *.ff text containing no special_bonds line returns Err
      whose message contains "special_bonds" (asserted by a test in
      molrs/src/ff/forcefield/readers/lammps.rs), and the identifiers AMBER_LJ14 and
      AMBER_COUL14 no longer appear anywhere in that file.
    status: pending
  - id: ac-005
    summary: Data-file coeff reads keep 0.5 / 0.8333 through an explicit declaration
    type: code
    pass_when: |
      LammpsFfReader::read_data_coeffs synthesizes the line
      "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333333334"; a test asserts the
      returned ForceField has lj_14() == 0.5 and |coul_14() - 5/6| < 1e-12, and the
      read_data_coeffs rustdoc states this is the data-file default the reader assumes.
    status: pending
  - id: ac-006
    summary: GROMACS writer emits [ defaults ] first with the force field's fudge factors
    type: code
    pass_when: |
      GromacsTopFfWriter::write_str output for special_bonds {lj:[0,0,0.5], coul:[0,0,5/6]}
      has "[ defaults ]" as its first section header and a data row whose whitespace-split
      tokens are exactly ["1", "2", "yes", "0.500000", "0.833333"]; asserted by a
      #[cfg(test)] test in molrs/src/ff/forcefield/writers/gromacs.rs.
    status: pending
  - id: ac-007
    summary: GROMACS reader parses [ defaults ] and rejects unsupported combinations
    type: code
    pass_when: |
      Tests in molrs/src/ff/forcefield/readers/gromacs.rs show read_str on
      "[ defaults ]\n1 2 yes 0.5 0.8333333333333334" gives lj[2] == 0.5 and
      |coul[2] - 5/6| < 1e-12, while nbfunc=2, comb-rule=3, gen-pairs=no, and a row missing
      fudgeQQ each return Err naming the offending field; a topology carrying a [ pairs ]
      section but no [ defaults ] returns Err whose message contains "defaults"; a fragment
      with neither section still parses and keeps SpecialBonds::default().
    status: pending
  - id: ac-008
    summary: XML writer takes 1-4 weights from ForceField, not from a pair-style default
    type: code
    pass_when: |
      Neither molrs/src/ff/forcefield/writers/xml.rs nor
      molrs/src/ff/forcefield/readers/opls.rs contains an unwrap_or default for
      coulomb14scale/lj14scale; a new #[cfg(test)] test writes a ForceField with coul_14 =
      1.0 and lj_14 = 1.0, reads it back with read_forcefield_xml_str, and gets 1.0 / 1.0
      (not 0.5), plus a 0.5 / 5/6 case round-tripping within 1e-6; and a test in
      readers/opls.rs asserts a <NonbondedForce> missing coulomb14scale (and one missing
      lj14scale) returns Err naming the attribute.
    status: pending
  - id: ac-009
    summary: AMBER 1-4 weights survive all three text formats within 1e-6
    type: scientific
    pass_when: |
      For SCEE = 1.2 and SCNB = 2.0 (weights 0.5 and 0.8333333333333334, Cornell 1995
      doi:10.1021/ja00124a002), each write-then-read round trip — LAMMPS *.ff, GROMACS
      [ defaults ], molrs XML <NonbondedForce> — returns lj_14 == 0.5 exactly and coul_14
      within 1e-6 of 5/6, asserted with hard-coded literals in the respective #[cfg(test)]
      modules. No LAMMPS, GROMACS, AmberTools or OpenMM binary is invoked by any test.
    status: pending
  - id: ac-010
    summary: Rustdoc states the new contracts and the behaviour break
    type: docs
    pass_when: |
      The module rustdoc of molrs/src/ff/forcefield/readers/lammps.rs no longer claims 1-4
      scaling "follows the AMBER/GAFF convention" and instead states that special_bonds must
      be declared and that an absent line is an Err; read_data_coeffs documents its
      synthesized declaration; the GROMACS reader module rustdoc documents [ defaults ]
      parsing, its three rejections, and the absent-section behaviour; the LAMMPS and GROMACS
      writer module rustdocs show the emitted line/section; the readers/opls.rs rustdoc states
      that coulomb14scale / lj14scale are required (absent -> Err, no 0.5 default). `cargo test
      --doc -p molcrafts-molrs --features full,filesystem` passes.
    status: pending
  - id: ac-011
    summary: Regression script proves the prmtop 1-4 weights reach both text formats
    type: runtime
    pass_when: |
      `python regressions/amber-prmtop-complete-03-lammps.py` exits 0 with no third-party
      scientific software installed. It reads its embedded prmtop literal via
      molrs.ff.read_amber_prmtop_ff_str, asserts the LAMMPS text carries
      "special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333" after
      `units real`, that a re-read/re-write reproduces that line byte-for-byte, that removing
      the line makes read_lammps_forcefield_str raise ValueError mentioning special_bonds,
      and that the GROMACS text's first section is [ defaults ] with tokens
      ["1","2","yes","0.500000","0.833333"] surviving a read/re-write (goldens 1/2.0 and
      1/1.2, tolerance 1e-6, hard-coded with their SCEE/SCNB provenance in comments).
    status: pending
  - id: ac-012
    summary: Full check and test suite green
    type: runtime
    pass_when: |
      `cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features
      full,filesystem -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features
      full,filesystem`, `cargo test --doc -p molcrafts-molrs --features full,filesystem`,
      and `uv --directory molrs-python run --no-sync tox -e py` all pass on the phase branch.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** are the architect's 🔴1: the line is unconditional, so no option and no binder
  churn. ac-002 is checkable by reading the struct and by `git diff --stat`.
- **ac-003 – ac-005** are the reader half of 🔴2. ac-004 is the behaviour break; ac-005 is the promise
  that LAMMPS DATA reads keep today's numbers through an *explicit* declaration.
- **ac-006 / ac-007** are the GROMACS pair, including 🟡5's symmetric `gen-pairs` validation. The
  absent-`[ defaults ]` clause in ac-007 pins the deliberate asymmetry with ac-004.
- **ac-008** is 🔴4: the `unwrap_or(0.5)` must be gone, and the `coul_14 = 1.0` case is what proves it
  (a surviving default would still pass a 0.5-only test).
- **ac-009** is the only `scientific` criterion — everything else is structural. It is verified inside
  the repo's own unit tests with literals, never against a third-party engine.
- **ac-011** is `runtime`, not `scientific`: the script lives in this repo's `regressions/` and
  reproduces hard-coded values, with no external bench repo involved.
