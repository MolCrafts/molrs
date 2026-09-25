---
spec: opls-gromacs-02-table
created: 2026-09-25
criteria:
  - id: ac-001
    summary: OplsAtomRow carries parameters only; OplsRuleRow holds the moved rules
    type: code
    pass_when: |
      In `molrs/src/ff/params/mod.rs`, `OplsAtomRow` has exactly name, class, mass,
      charge, sigma, epsilon; `OplsRuleRow` has name, def, overrides, and its doc comment
      and `OplsTypeRow`'s (`ff/typifier/opls/meta.rs`) each state the static-rule →
      runtime-record relationship; `oplsaa_typing.rs` defines `OPLSAA_TYPING` with 157
      rows whose def and overrides equal the pre-link values byte for byte.
    status: verified
    last_checked: 2026-09-25
    note: verbatim except opls_927 drops its override of opls_928 (no rule; could never apply) — recorded in the oplsaa_typing.rs header; opls-gromacs-03 decides whether opls_928 gets a rule
  - id: ac-002
    summary: The generator is an unpublished example run through a build-set-1 alias
    type: code
    pass_when: |
      `molrs/Cargo.toml` declares `[[example]] name = "gen_opls_params"` with
      `required-features = ["ff"]`, `exclude = ["examples/"]`, and `sha2` under
      `[dev-dependencies]` only; `.cargo/config.toml` defines `mrs-gen-opls` ending in
      `"--"` with feature string `full,filesystem,stream`; the legitimate-builds text in
      `.cargo/config.toml` and `.claude/notes/build.md` names the example unit; CLAUDE.md
      § Build & Test Commands lists `cargo mrs-gen-opls --gromacs <dir>`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-003
    summary: One command reproduces oplsaa.rs byte for byte from the pinned files
    type: runtime
    pass_when: |
      With GROMACS v2026.3 (`42105e4672205b4aa951962b8e3cdb4c27890da1`)
      `share/top/oplsaa.ff` present, `cargo mrs-gen-opls --gromacs <dir>` exits 0 and
      `git diff --exit-code molrs/src/ff/params/oplsaa.rs` reports no change; a modified
      input file makes it exit non-zero naming the SHA-256 mismatch.
    status: verified
    last_checked: 2026-09-25
  - id: ac-004
    summary: The generator adds no library fingerprint and is not packaged
    type: runtime
    pass_when: |
      After `cargo mrs-build` then `cargo mrs-gen-opls --gromacs <dir>`, the set of
      `target/debug/.fingerprint/molcrafts-molrs-*` directories containing a
      `lib-molrs*` file is unchanged and the only new such directory contains
      `example-gen_opls_params*`; `cargo package --list -p molcrafts-molrs` lists no
      path under `examples/`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-005
    summary: The table header is the complete provenance record
    type: docs
    pass_when: |
      The `oplsaa.rs` header states: DO-NOT-HAND-EDIT with the command
      `cargo mrs-gen-opls --gromacs <path>`; tag, commit, the three paths and SHA-256s;
      LGPL-2.1-or-later attribution; unit conversions; rows read per section and rows
      emitted; and that `[ constrainttypes ]`, the six improper macros and `at.num` are
      not encoded, each with its reason.
    status: verified
    last_checked: 2026-09-25
  - id: ac-006
    summary: Hand-converted GROMACS rows match the committed table
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::typifier::opls::embedded` passes pins: opls_135 (CT, 12.011,
      -0.18, 3.5, 0.066); opls_150 ("C=", -0.115, 3.55, 0.076); opls_155 sigma 0.0; bond
      CT-HC (1.09, 680.0); bond C=-C= (1.46, 770); angle CM-C=-C= (124°·π/180, 140);
      dihedral HC-CT-CT-HC (0, 0, 0.3, 0); relative tolerance 1e-12.
    status: verified
    last_checked: 2026-09-25
  - id: ac-007
    summary: Embedded lj/cut declares geometric mixing
    type: runtime
    pass_when: |
      An embedded test asserts the `pair/lj/cut` style of
      `OPLSAATypifier::oplsaa().library()` has string param `mixing == "geometric"`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-008
    summary: Typing metadata joins every rule to a GROMACS-classed atom row
    type: runtime
    pass_when: |
      `ff::typifier::opls::embedded::tests::typing_meta_joins_every_rule` asserts
      `try_typing_meta()` is Ok and opls_135's class is "CT"; test-local tables with a
      rule naming no atom row, or an override naming no rule, are Err; `typing_meta()`'s
      expect message names that test path.
    status: verified
    last_checked: 2026-09-25
  - id: ac-009
    summary: LAMMPS export states the mixing rule the kernel uses
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::forcefield::writers::lammps` passes tests that an lj/cut
      without `mixing` writes `pair_modify mix arithmetic` and one with `geometric`
      writes `pair_modify mix geometric`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-010
    summary: Foyer-only rows are gone and the GROMACS-only row is present
    type: code
    pass_when: |
      `oplsaa.rs` contains no angle rows O_3-C-O_3 or OS-C_2-OS, no dihedral rows
      NZ-CZ-CA-X or CT-OS-C_2-OS, and contains the dihedral row H-N-CT_2-C.
    status: verified
    last_checked: 2026-09-25
  - id: ac-011
    summary: Notes record the typing_meta amendment and the mandatory-mixing follow-up
    type: docs
    pass_when: |
      notes.md's scoped-amendment list names `typing_meta` → `try_typing_meta` with its
      test path, and notes.md holds a routed `/mol:spec` entry "mixing mandatory on every
      lj/cut" citing `ff/potential/pair/lj_cut.rs:644-648`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-012
    summary: Doctest regression example on OPLSAATypifier::oplsaa
    type: runtime
    pass_when: |
      A rustdoc example on `OPLSAATypifier::oplsaa` asserts the library's lj/cut
      `mixing` equals "geometric"; it passes under `cargo mrs-doctest` in the link-03
      chain-end gate.
    status: pending
    note: chain-end gate
  - id: ac-013
    summary: Unit suite green at link close; full gate discharged at chain end
    type: runtime
    pass_when: |
      `cargo mrs-test` exits 0 at link close; the full gate is discharged by link 03's
      `prek run --all-files --hook-stage pre-push`.
    status: pending
    note: chain-end gate
---

# Acceptance criteria

ac-003 is reproducibility, and ac-004 is the build-cache check the architect asked for. ac-006 pins numbers against hand-converted source lines, never against another program's output. ac-009 fixes the kernel/LAMMPS disagreement for every force field that declares no rule.
