---
slug: amber-prmtop-complete-02-forcefield
criteria:
  - id: ac-001
    summary: prmtop reader emits registered lj/cut + coul/cut pair styles
    type: code
    pass_when: |
      Reading the inline GAFF_MINI fixture through AmberPrmtopFfReader yields a
      ForceField where get_style("pair","lj/cut") and get_style("pair","coul/cut")
      both resolve, get_style("pair","lj/cut/coul/long") is None, and every style
      name used is present in KernelRegistry::builtin().
    status: pending
  - id: ac-002
    summary: coul/cut declares coulomb, dielectric and cutoff explicitly
    type: code
    pass_when: |
      The coul/cut style's params contain coulomb = 332.05221729,
      dielectric = 1.0 (VACUUM_DIELECTRIC) and cutoff = 10.0, and the lj/cut
      style's params contain cutoff = 9.0; no pair style in the returned
      ForceField carries a cutoff_lj or cutoff_coul key.
    status: pending
  - id: ac-003
    summary: AMBER constants have one crate-internal home and the documented value
    type: scientific
    pass_when: |
      `rg -n "AMBER_COULOMB\s*:\s*f64\s*=" molrs/src` matches exactly once, in
      the new hand-maintained molrs/src/ff/params/amber.rs, as do AMBER_SCEE and
      AMBER_SCNB; params/amber.rs is NOT listed in molrs/src/ff/params/MANIFEST.sha256
      and the generated params/gaff.rs / gaff2.rs are unchanged; both
      molrs/src/ff/forcefield/gaff.rs and readers/prmtop.rs import them (no
      AMBER_LJ_14 / AMBER_COUL_14 / DEFAULT_SCEE / DEFAULT_SCNB definitions
      remain); a unit test asserts |AMBER_COULOMB - 18.2223_f64.powi(2)| < 1e-9
      and |(COULOMB_REAL - AMBER_COULOMB)/COULOMB_REAL - 3.4610e-5| < 1e-8; the
      AMBER_COULOMB rustdoc carries both provenances (18.2223 charge factor with
      the ambermd.org / ParmEd reference, and the sander single-point
      measurement, with no reference to the non-existent
      scripts/gen_gaff_energy_oracle.py); and the AMBER_SCEE / AMBER_SCNB rustdoc states both roles
      (GAFF typifier 1-4 parameter; prmtop absent-section fallback).
    status: pending
  - id: ac-004
    summary: written pair_coeff block is byte-identical to pre-change output
    type: code
    pass_when: |
      Test pair_coeff_text_is_pinned compares LammpsFfWriter output for the
      GAFF_MINI fixture against a hard-coded literal captured from the
      pre-change reader and the pair_coeff lines match byte for byte, including
      order; test litfsi_corpus_pair_coeff_text_is_pinned does the same for the
      committed tests-data/prmtop/LiTFSI.prmtop (seven pair_coeff lines, one per
      distinct AMBER_ATOM_TYPE name, first-appearance order) and
      asserts lj_14 == 0.5, coul_14 == 1.0/1.2 on that real file.
    status: pending
  - id: ac-005
    summary: skip_pair_style include still contains only pair_coeff lines
    type: code
    pass_when: |
      With LammpsWriteOptions { skip_pair_style: true, .. } the written string
      for the GAFF_MINI-derived force field contains no "pair_style" substring
      and still contains every pinned "pair_coeff <t> <t> <eps> <sigma>" line.
    status: pending
  - id: ac-006
    summary: uniform_divisor defaults, rejects non-uniform, ignores suppressed 1-4
    type: code
    pass_when: |
      uniform_divisor returns 1.2/2.0 when the SCEE/SCNB sections are absent;
      returns Err containing "SCEE_SCALE_FACTOR" and both divisor values when two
      non-suppressed torsions reference divisors differing by more than 1e-6
      relative; returns Ok when the divergent divisor is only referenced by a
      torsion with a negative 3rd pointer; and the resulting SpecialBonds carry
      lj[2] = 1/SCNB and coul[2] = 1/SCEE.
    status: pending
  - id: ac-007
    summary: decode_lj_types self terms reproduce the A/B closed form
    type: scientific
    pass_when: |
      For LENNARD_JONES_ACOEF = 1043080.23 and BCOEF = 675.612248 the decoded c3
      self term gives epsilon = 0.109400 kcal/mol and sigma = 3.3996695 A, both
      within 1e-6 relative of those hard-coded values.
    status: pending
  - id: ac-008
    summary: LB-consistent cross entries pass; NBFIX-deviating ones are refused
    type: scientific
    pass_when: |
      An exact-LB c3/hc off-diagonal ICO entry yields Ok with no two-endpoint
      pair type in the ForceField, while the same entry with epsilon scaled by
      1.01 yields Err whose message contains "c3", "hc" and "not supported"; no
      call to def_pairtype(_, Some(_), _) exists in readers/prmtop.rs.
    status: pending
  - id: ac-009
    summary: unsupported prmtop features are named errors, never silent drops
    type: code
    pass_when: |
      A fixture with LENNARD_JONES_CCOEF returns Err containing "12-6-4"; a
      multi-term improper returns Err containing its dihedral type id; a negative
      ICO entry and a non-zero HBOND_ACOEF/HBOND_BCOEF each return
      Err("10-12 interactions are not supported").
    status: pending
  - id: ac-010
    summary: no new public symbol and no with_cutoffs builder
    type: code
    pass_when: |
      The public surface of molrs::ff::forcefield::readers::prmtop after the
      change is exactly AmberPrmtopFfReader (with new + ForceFieldReader impl)
      and read_amber_prmtop_ff with its current signature; with_cutoffs does not
      exist; uniform_divisor / decode_lj_types / the LjSelfRow alias / the two
      cutoff constants are private; and the three AMBER constants in
      params/amber.rs are pub(crate), not pub.
    status: pending
  - id: ac-011
    summary: rustdoc debt notes and stale-doc fixes are in place
    type: docs
    pass_when: |
      A "Debt:" rustdoc note naming the duplicated sigma/epsilon closed form and
      its revisit trigger appears at both readers/prmtop.rs and
      io/data/prmtop_tables.rs::decode_nonbond_params; the same debt and the
      molrs/tests/architecture_gate.rs-vs-"no tests/ tree" documentation drift
      each have a dated entry in .claude/notes/notes.md; .claude/notes/release.md carries a bullet
      under the next untagged version naming the pair-style split, the
      lj/cut/coul/long 10 10 -> lj/cut/coul/cut 9 10 header change, and the new
      refusals; kspace/pme.rs lines 10 and 857 say atomi/atomj;
      io/data/prmtop.rs:47 states 18.2223 with 18.2223^2 = 332.05221729; and
      `cargo test --doc -p molcrafts-molrs --features full,filesystem` passes.
    status: pending
  - id: ac-012
    summary: regression example reproduces the hard-coded prmtop force-field goldens
    type: runtime
    pass_when: |
      `python regressions/amber-prmtop-complete-02-forcefield.py` exits 0, having
      asserted from molrs.ff.read_amber_prmtop_ff_str alone: c3 sigma = 3.3996695
      A and epsilon = 0.1094 kcal/mol (rel 1e-6), coul/cut coulomb =
      332.05221729, dielectric = 1.0, cutoff = 10.0, lj/cut cutoff = 9.0, and a
      "pair_coeff c3 c3" line with no "pair_style" substring in
      write_lammps_forcefield_str(ff, skip_pair_style=True); the file imports no
      AmberTools/ParmEd and spawns no subprocess.
    status: pending
  - id: ac-013
    summary: full check and test suite green
    type: code
    pass_when: |
      `cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets
      --features full,filesystem -- -D warnings` and `cargo test -p
      molcrafts-molrs --lib --features full,filesystem` all pass.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** close architect finding 🔴 1: the reader's pair styles must be buildable by `to_potentials` and must state `coulomb`/`dielectric` (CLAUDE.md:304), following `readers/lammps.rs::build_pairs`.
- **ac-003** pins the science of the constant choice — Amber's implied `18.2223²`, with the CODATA offset recorded rather than hidden.
- **ac-004 / ac-005** are the downstream hard constraint: the production PEO pipeline's LAMMPS include must not move a single character.
- **ac-006 / ac-007 / ac-008 / ac-009** cover the three silent behaviours this spec removes (arbitrary divisor pick, dropped off-diagonal LJ, dropped improper terms) plus the two refusals.
- **ac-010** closes architect finding 🔴 2 (`with_cutoffs` has no in-tree caller) and keeps Shape check #3 satisfied.
- **ac-011** closes architect finding 🟡 3 (accepted duplication, named at both sites) and the two Iron-law doc defects found in the touched surface.
- **ac-012** is verified by `/mol:impl` at delivery; the goldens are literals in this repo's `regressions/` script, never produced by a third-party tool at runtime.
