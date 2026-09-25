---
spec: opls-gromacs-01-gromacs-io
created: 2026-09-25
criteria:
  - id: ac-001
    summary: RB↔Fourier conversions live once, in ff::forcefield::torsion
    type: code
    pass_when: |
      `molrs/src/ff/forcefield/torsion.rs` defines `pub(crate) fn rb_to_opls([f64; 6])
      -> Result<[f64; 4], String>`, `pub(crate) fn opls_to_rb([f64; 4]) -> [f64; 6]` and
      `RB_TOL` documented as 1e-4 kJ/mol with the GROMACS 5-decimal rationale.
    status: verified
    last_checked: 2026-09-25
  - id: ac-002
    summary: No private conversion copy remains in the OPLS reader or the XML writer
    type: code
    pass_when: |
      Neither `readers/opls.rs` nor `writers/xml.rs` defines `fn rb_to_opls` or
      `fn opls_to_rb`; `writers/xml.rs` keeps its c0..c5 pass-through branch and calls
      `torsion::opls_to_rb` for the k1..k4 branch after converting kcal→kJ.
    status: verified
    last_checked: 2026-09-25
  - id: ac-003
    summary: Torsion conversions reproduce the hand-derived H-C-C-H row and refuse offsets
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::forcefield::torsion` passes, asserting
      rb_to_opls([0.62760,1.88280,0,-2.51040,0,0]) == [0,0,1.2552,0] (abs 1e-12),
      opls_to_rb of it returns the inputs, and Err for C5 = 0.1 and for [1,0,0,0,0,0].
    status: verified
    last_checked: 2026-09-25
  - id: ac-004
    summary: Mixing names the undeclared rule once, crate-internally
    type: code
    pass_when: |
      `Mixing::UNDECLARED` (== Arithmetic) and `Mixing::name` are `pub(crate)`;
      `pair_lj_cut_ctor` uses `UNDECLARED` for the absent case; the unknown-rule message
      in `mixing.rs` has no run of two or more spaces.
    status: verified
    last_checked: 2026-09-25
  - id: ac-005
    summary: Reader maps comb-rule 2/3 to mixing; refuses comb-rule 1 and nbfunc 2
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::forcefield::readers::gromacs` passes with tests asserting
      comb-rule 3 → lj/cut mixing "geometric", 2 → "arithmetic", and Err naming the value
      for comb-rule 1 and nbfunc 2.
    status: verified
    last_checked: 2026-09-25
  - id: ac-006
    summary: [ atomtypes ] splits into atom/full and the lj/cut self row
    type: runtime
    pass_when: |
      A reader test on the opls_135 8-column row asserts atom/full mass 12.011, charge
      -0.18, atomic_number 6, bond_type "CT", ptype "A", no sigma/epsilon on the atom
      type, and an lj/cut self row with sigma 3.5 and epsilon 0.066 (abs 1e-12); a
      pair/coul/cut style declares coulomb and dielectric.
    status: verified
    last_checked: 2026-09-25
  - id: ac-007
    summary: Bonded directives convert to molrs kernels with hand-derived values
    type: runtime
    pass_when: |
      Reader tests assert: CT-HC → bond/harmonic (1.09, 680.0); HC-CT-HC →
      angle/harmonic (107.8·π/180, 66.0); funct 3 HC-CT-CT-HC → dihedral/opls
      (0,0,0.3,0); funct 1 → dihedral/periodic; funct 4 → improper/periodic (2.5, π);
      funct 2 with ξ0 = 0 → improper/harmonic K 20.0; bond funct 3 → bond/morse
      (95.6022944…, 2.0, 1.529); GROMACS `X` → empty endpoint.
    status: verified
    last_checked: 2026-09-25
  - id: ac-008
    summary: Unmodelled codes, sections and molecule sections are refused by name
    type: runtime
    pass_when: |
      Reader tests assert Err naming the cause for dihedral funct 9, funct 2 with ξ0 ≠ 0,
      a funct-3 row with ΣC ≠ 0, bond funct 2, angle funct 5, `[ pairtypes ]`,
      `[ constrainttypes ]`, `[ atoms ]` (message names `read_top`), `#if`; and Ok for
      `[ constrainttypes ]` after `with_skipped_directive("constrainttypes")`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-009
    summary: Preprocessor conditionals follow the define set
    type: runtime
    pass_when: |
      Reader tests assert rows in `#ifdef FOO … #endif` are read iff `#define FOO`
      precedes, `#ifndef` is the complement, and `#else` selects the other branch.
    status: verified
    last_checked: 2026-09-25
  - id: ac-010
    summary: The reader is directive-only with private configuration; free doors gone
    type: code
    pass_when: |
      `readers/gromacs.rs` has no `[ atoms ]`-index resolution code, no `pub` fields on
      `GromacsTopFfReader`, builders `with_include` and `with_skipped_directive` only (no
      `include_dirs` capability), and no `fn read_gromacs_top_ff`; `writers/gromacs.rs`
      has no `fn write_gromacs_top_ff` / `write_gromacs_top_ff_str`; `ff/mod.rs`
      re-exports none of the three.
    status: verified
    last_checked: 2026-09-25
  - id: ac-011
    summary: Writer emits directives only and never invents atomtype values
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::forcefield::writers::gromacs` passes asserting: no
      `[ atoms ]`/`[ bonds ]`/`[ angles ]`/`[ dihedrals ]`/`[ pairs ]` in output;
      geometric → `1  3  yes`, undeclared → `1  2  yes`; atomtypes sigma/epsilon from the
      lj/cut self row; dihedral/opls k3 0.3 → `3` with 0.6276 1.8828 0 -2.5104 0 0; empty
      endpoint → `X`; Err naming the cause for an atom type lacking mass, lacking
      charge, or lacking its lj/cut self row, and for sixthpower, dihedral/charmm, a
      multi-term periodic, an explicit lj/cut cross row, a non-zero 1-2 special-bond
      weight, and an unresolvable bonded endpoint label.
    status: verified
    last_checked: 2026-09-25
  - id: ac-012
    summary: GROMACS write then read is the identity on supported styles
    type: runtime
    pass_when: |
      A writer test writes a ForceField holding every supported style, reads it back with
      GromacsTopFfReader, and asserts equal styles, type names, endpoints and params
      (abs 1e-9).
    status: verified
    last_checked: 2026-09-25
  - id: ac-013
    summary: XML reader refuses non-representable RB rows; XML writer uses torsion.rs
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::forcefield::readers::opls ff::forcefield::writers::xml`
      passes, including an Err test for an RBTorsionForce row with c5 ≠ 0 and a writer
      test that k3 0.3 kcal/mol is written as c0..c5 = 0.6276, 1.8828, 0, -2.5104, 0, 0.
    status: verified
    last_checked: 2026-09-25
  - id: ac-014
    summary: Rustdoc and notes state the model, and route the found debt
    type: docs
    pass_when: |
      Module rustdoc of `readers/gromacs.rs` / `writers/gromacs.rs` lists supported
      directives, the funct → style map with units, refusals, the skip builder and the
      molecule-section refusal; `torsion.rs` documents both relations, representability
      and RB_TOL units; notes.md's 2026-09-25 GROMACS entry records the molecule model
      removed, and notes.md holds routed entries with path:line for the OplsXmlReader
      drops (readers/opls.rs:137-146, :457, :335-360; /mol:fix), io/data/top.rs:271-273
      (/mol:fix), the duplicated unit constants (/mol:refactor), and OPLS improper
      assignment (/mol:spec), and the remaining Rust convenience free functions
      (`ff/mod.rs:13-26`, `writers/xml.rs:348`; /mol:refactor).
    status: verified
    last_checked: 2026-09-25
  - id: ac-015
    summary: Doctest regression example on GromacsTopFfReader
    type: runtime
    pass_when: |
      A rustdoc example on `GromacsTopFfReader` reads an inline string with
      `[ defaults ] 1 3 yes 0.5 0.5`, one atomtypes row and one bondtypes row, and
      asserts mixing == "geometric" and the bond r0 in Å against literals; it passes
      under `cargo mrs-doctest` in the link-03 chain-end gate.
    status: pending
    note: chain-end gate
  - id: ac-016
    summary: Unit suite green at link close; full gate discharged at chain end
    type: runtime
    pass_when: |
      `cargo mrs-test` exits 0 at link close; the full gate is discharged by link 03's
      `prek run --all-files --hook-stage pre-push`.
    status: pending
    note: chain-end gate
---

# Acceptance criteria

ac-001 and ac-002 close the duplicated conversion. ac-005 through ac-011 are the reader and writer contracts. ac-010 pins the directive-only shape, and ac-011 pins "no invented values". ac-014 is how the rot this link found gets routed rather than left silent.
