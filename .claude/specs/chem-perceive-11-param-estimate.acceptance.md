---
slug: chem-perceive-11-param-estimate
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Native estimation reproduces parmchk2 on the whole oracle
    type: code
    pass_when: |
      For all 37 molecules, every term parmchk2 emits into its frcmod is reproduced natively:
              same fallback tier (PenaltyTier) and same value within table precision. The RED fixture
              from spec 10 is green.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `molrs/src/ff/typifier/estimate/parmchk.rs` (806 lines). All four `parmchk2_oracle` tests green
      across 37 molecules x {gaff, gaff2}: the estimated-term SET, every value, every penalty tier,
      and full parameterisation of every molecule. Two halves were both required — wildcard row
      matching (which silently resolves ~145 terms/molecule that an exact-match scan calls "missing")
      and analogy+penalty scoring for the rest.
  - id: ac-002
    summary: No frcmod file I/O remains
    type: code
    pass_when: |
      `grep -rniE 'frcmod' molrs/src molrs/tests` returns 0 hits outside comments. No test reads an
              external .frcmod file. molrs does not depend on any external parameter file at runtime.
              AMENDED 2026-07-14: originally `grep -rnE 'Frcmod::parse_str|write_string' molrs/src`.
              The bare `write_string` alternative also matches `io/trajectory/{xdr,trr}.rs` — XDR's
              string writer, which has nothing to do with frcmod and predates this chain. The
              amended grep is *stricter*: it demands the whole frcmod concept be gone, not just its
              two entry points.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `molrs/src/ff/frcmod.rs` deleted outright (379 lines) along with its `ff/mod.rs` re-export and
      both test files. The estimator produces `ForceField` terms in memory; nothing serialises to or
      parses the frcmod text format.
  - id: ac-003
    summary: The estimation is built on the existing estimator architecture
    type: code
    pass_when: |
      The GAFF fallback goes through the single estimator / ParameterInterpolator /
              TypifierParameterContext — not a parallel reimplementation. The OPLS context still works
              unchanged (its existing tests stay green).
    status: verified
    last_checked: 2026-07-14
    evidence: |
      VIOLATED on the first pass and reworked. That pass added a second, parallel `ParmchkEstimator`
      stack (`estimate/parmchk.rs`, 806 lines) beside the existing estimator; the owner rejected it.
      Now there is exactly ONE trait (`ParameterInterpolator`) and ONE implementor, `Parmchk2Estimator`
      — the old `ParameterEstimator` renamed, since it IS parmchk2's algorithm and an abstract name on a
      concrete algorithm was itself the smell. `estimate/parmchk.rs` is deleted; its algorithm became the
      shared cascade. Both force fields enter through it: OPLS `::new(ff, meta)` (`opls/mod.rs:141` +
      12 tests in `tests/ff/typifier/estimate.rs`, all green), GAFF `::with_context(ff, gaff_ctx)`.
      `BondedTerm` moved out of `typifier/opls/` up into `estimate/term.rs` and gained a fourth,
      co-equal `Improper` variant. No `ParameterEstimator` alias survives.
  - id: ac-004
    summary: Fallback tiers are ordered exactly as parmchk2
    type: code
    pass_when: |
      A test drives a term through each tier in turn: exact match, wildcard (X) row,
              atom-type-equivalence substitution (corr row, with penalty), empirical formula —
              and asserts the chosen tier matches parmchk2's for that term.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Inline tier tests on the one cascade: exact; wildcard row (`X -c3-c3-X ` covers ethane's
      hc-c3-c3-hc → `Estimate::Covered`, NOT an estimate, no penalty); equivalent-type substitution
      (gaff2 `ns`→`n`, penalty 0.0); corresponding-type substitution (thiophene `cc-cd-ss-cd` off
      `X -c2-ss-X `, penalty 232.0 — matches parmchk2); improper off a wildcard row (benzene
      `ca-ca-ca-ha` off `X -X -ca-ha`, penalty 6.0); default improper; and the empirical tier.
      The empirical case is H-Br: an earlier attempt used `br-br`, which silently passed through the
      EXACT tier because gaff.dat has that row — H bonded to Br is the genuine hole (no bond row for
      that element pair, so no analogy is reachable, while the element-keyed PARM_BLBA_GAFF.DAT does
      carry it). The empirical tier remains UNREACHABLE from the 37-molecule oracle (it estimates only
      torsions), and the code comment saying so is preserved. Not claimed as oracle-validated.
---
