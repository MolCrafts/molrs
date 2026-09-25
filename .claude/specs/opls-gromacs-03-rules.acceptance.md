---
spec: opls-gromacs-03-rules
created: 2026-09-25
criteria:
  - id: ac-001
    summary: Dominance ranks candidates pairwise and order-independently
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::typifier::opls::layered` passes tests asserting: a
      (transitively) overriding candidate wins even when less specific; a higher-layer
      candidate beats a lower-layer one regardless of overrides; non-dominated candidates
      rank by explicit priority, then query-atom count, then earlier name.
    status: verified
    last_checked: 2026-09-25
  - id: ac-002
    summary: A later level never replaces a type that dominates the new candidate
    type: runtime
    pass_when: |
      A layered.rs test with level-0 T0 overriding level-1 T1 on one atom asserts T0 is
      kept; without that override T1 replaces T0.
    status: verified
    last_checked: 2026-09-25
  - id: ac-003
    summary: Override cycles and dangling overrides are build errors on every path
    type: runtime
    pass_when: |
      Layered.rs tests assert `LayeredTypingEngine::build` is Err naming the members for
      an overrides cycle, and Err naming both types for an override naming a type absent
      from the metadata; `OPLSAATypifier::from_xml_str` on XML with a dangling `overrides`
      attribute returns Err at construction (no typifier is built).
    status: verified
    last_checked: 2026-09-25
  - id: ac-004
    summary: The derived priority score is gone
    type: code
    pass_when: |
      `OplsTypingMeta::priorities` and `LAYER_PRIORITY_STRIDE` do not exist in
      `molrs/src` and are not re-exported from `ff/typifier/opls/mod.rs`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-005
    summary: Typing perceives aromaticity on a private copy
    type: runtime
    pass_when: |
      A typing.rs test types Kekulé benzene with `c`-only rules (12 atoms typed) and
      asserts the returned graph's ring bonds keep their input Double/Single orders.
    status: verified
    last_checked: 2026-09-25
  - id: ac-006
    summary: Any type name is a context-label dependency
    type: runtime
    pass_when: |
      A deps.rs test asserts `[#1]-[#6;%CX]` is at level 1 when `CX` is a type with no
      `opls_` prefix; the `starts_with("opls_")` filter is absent.
    status: verified
    last_checked: 2026-09-25
  - id: ac-007
    summary: The rules table follows the Daylight conventions
    type: code
    pass_when: |
      Every `def` in `oplsaa_typing.rs` writes each inter-atom bond with `-`, `=`, `#`,
      `:` or a commented `~`; hydrogen atoms appear only as `[#1]`; `[Cl,C,H]` and `[!H]`
      do not occur; monatomic-ion rules carry `X0`; the header states the conventions and
      lists changed overrides.
    status: verified
    last_checked: 2026-09-25
  - id: ac-008
    summary: Diene types opls_150 and opls_178 exist with their overrides
    type: code
    pass_when: |
      `OPLSAA_TYPING` contains opls_150 `[C;X3;H1](=[C;X3])-[C;X3]=[C;X3]` overriding
      opls_142 and opls_178 `[C;X3;H0](=[C;X3])(-[#6])-[C;X3]=[C;X3]` overriding opls_141.
    status: verified
    last_checked: 2026-09-25
  - id: ac-009
    summary: Fifteen golden molecules type exactly and are neutral
    type: runtime
    pass_when: |
      `cargo mrs-test -- ff::typifier::opls::embedded` passes golden tests for NMA,
      1,3-butadiene, ethanol, benzene (aromatic and Kekulé), propylene carbonate, methyl
      methacrylate, methyl formate, benzonitrile, chlorobenzene, chloroethane,
      fluorobenzene, pyridine, pyrimidine and pyrrole, each asserting every atom's type
      from the spec table and |Σ type charges| < 1e-9.
    status: verified
    last_checked: 2026-09-25
  - id: ac-010
    summary: Strict-refusal tests use a molecule the rules cannot cover
    type: runtime
    pass_when: |
      `opls/mod.rs` tests build methylsilane (C 0, Si 1, H-on-C 2..=4, H-on-Si 5..=7);
      strict typing is Err naming atoms 0, 1, 5, 6, 7 and none of 2..=4; non-strict is
      Ok; no test uses butadiene as an untyped example.
    status: verified
    last_checked: 2026-09-25
  - id: ac-011
    summary: Rustdoc describes Daylight rules, perception and dominance
    type: docs
    pass_when: |
      `typing.rs`'s conflict-resolution section describes dominance then
      priority/size/name; it and `opls/mod.rs` state private-copy perception;
      `OplsRuleRow` states Daylight SMARTS with explicit hydrogens.
    status: verified
    last_checked: 2026-09-25
  - id: ac-012
    summary: Python readers expose skip_directives; writers call the writer type
    type: code
    pass_when: |
      In `molrs-python/src/ff/mod.rs`, `read_gromacs_top_ff` and
      `read_gromacs_top_ff_str` have signature `(…, include = false, *, skip_directives =
      ())` and apply each via `with_skipped_directive`; the write functions call
      `GromacsTopFfWriter`; no Rust path `molrs::ff::read_gromacs_top_ff` or
      `molrs::ff::write_gromacs_top_ff*` remains; the public wrappers in
      `molrs-python/python/molrs/ff/forcefield.py` take keyword-only `skip_directives`
      and pass it through; the wrapper docstrings, `_lib` docstrings and `_lib.pyi`
      entries describe the directive model and `skip_directives`.
    status: verified
    last_checked: 2026-09-25
  - id: ac-013
    summary: One Python seam smoke test covers the skip keyword
    type: runtime
    pass_when: |
      `molrs-python/tests/test_forcefield_gromacs_reader.py` asserts
      the public `molrs.ff.read_gromacs_top_ff_str` raises ValueError on a string with
      `[ constrainttypes ]`
      and returns a ForceField when `skip_directives=["constrainttypes"]`; it passes in the
      tox run of the chain-end gate.
    status: pending
    note: chain-end gate
  - id: ac-014
    summary: Doctest regression example types ethanol's oxygen
    type: runtime
    pass_when: |
      A rustdoc example on `OPLSAATypifier::oplsaa` types hand-built ethanol and asserts
      the oxygen's type equals "opls_154"; it passes under `cargo mrs-doctest` in this
      link's gate.
    status: pending
    note: chain-end gate
  - id: ac-015
    summary: Chain-end full gate green, discharging links 01-03
    type: runtime
    pass_when: |
      After committing (pre-commit rustfmt + clippy pass),
      `prek run --all-files --hook-stage pre-push` exits 0 on the tree containing links
      01-03 — covering fmt, `cargo mrs-clippy -- -D warnings`, `cargo mrs-test`,
      `cargo mrs-doctest`, rustdoc and every binder build/test — which discharges the
      full-gate criterion of opls-gromacs-01, -02 and -03.
    status: pending
    note: chain-end gate
---

# Acceptance criteria

ac-001 through ac-006 are the engine's contract. ac-003 is the architect's "dangling override is an error" ruling. ac-009 is the chemistry contract: per-atom hand-derived types plus the neutrality invariant. ac-012 and ac-013 restore Python↔Rust reader symmetry, and they are the chain's only binder criteria. ac-015 is the chain's single full gate.
