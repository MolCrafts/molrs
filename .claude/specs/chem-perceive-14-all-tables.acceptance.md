---
slug: chem-perceive-14-all-tables
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Every parameter table is a committed .rs
    type: code
    pass_when: |
      mmff94, mmff94s and oplsaa are committed typed Rust tables. `grep -rn 'include_str!' molrs/src`
              returns 0 hits. `molrs/data/*.xml` no longer exists. The unused molrs/data/gen3d fragment
              library no longer exists.
              ONE PLACE, ONE FORM (owner's ruling): every parameter table lives directly under
              `molrs/src/ff/params/`. There is NO directory named `generated` — these tables are
              first-class source, not a build artefact, and "generated" names how they ARRIVED, not what
              they ARE (their provenance belongs in each file's header doc). `molrs/src/ff/mmff/tables.rs`
              (51,621 lines, ported from RDKit Params.cpp) moves to `molrs/src/ff/params/mmff.rs` with its
              17 binary-search accessors — `git mv` plus import fixes, not one line of logic changed. So
              `molrs/src/ff/params/generated/` must not exist, and `molrs/src/ff/mmff/tables.rs` must not
              exist.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-002
    summary: Pure representation change — zero numerical delta
    type: code
    pass_when: |
      The MMFF94/MMFF94s/OPLS typifier and potential test suites and the conformer/ETKDG tests
              all pass with NO numerical assertion change caused by the table representation update.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-003
    summary: Generator byte-reproduces every table
    type: code
    pass_when: |
      Byte-for-byte regeneration is validated EXACTLY ONCE, locally, during implementation: with
              $AMBERHOME set, re-running `scripts/gen_param_tables.py` leaves `git diff --exit-code` clean.
              The measured result is recorded in the spec body.
              CI HAS ZERO COUPLING TO AMBERHOME (owner's ruling: "ci 不要和 AMBERHOME 有任何牵连，
              只在实施过程中验证一次"). Therefore `grep -rn AMBERHOME molrs/tests` returns 0 hits — every
              $AMBERHOME-dependent test is DELETED from the suite, not skipped. A test that skips itself in
              CI is a test that never runs; it buys the appearance of coverage and none of the substance.
              AMENDED 2026-07-14: the original wording ("Skips cleanly when sources are unavailable")
              mandated exactly the skip-in-CI behaviour the owner then ruled out.
    status: pending
    last_checked: 
    evidence: 
  - id: ac-004
    summary: Final binary size and build time are measured and recorded
    type: runtime
    pass_when: |
      The AFTER numbers are measured with the SAME commands as the BEFORE numbers and recorded in the
              spec body. BEFORE (measured 2026-07-14, release, --features full):
                clean build .............. 36.18 s
                libmolrs.rlib ............ 34,949 KB
                example binary (stripped) . 1,885 KB   [target/release/examples/typify_molecule]
              Reproduce with:
                cargo clean -p molcrafts-molrs
                /usr/bin/time -p cargo build --release -p molcrafts-molrs --features full
                cargo build --release -p molcrafts-molrs --features full --example typify_molecule
                strip -o /tmp/tm_after target/release/examples/typify_molecule && ls -la /tmp/tm_after
              The net binary delta is expected to be SMALL or negative (the raw-text copies and their
              runtime XML parsers are removed). Any regression beyond +2 MB must be justified in writing,
              not waved through.
              AMENDED 2026-07-14: type changed `performance` -> `runtime`. A `performance` criterion parks
              the spec at code-complete owing an evaluator that does not exist here; this is a plain
              measurement with two commands, so it is checkable in-band.
    last_checked: 
    evidence: 
---
