---
name: molrs-debug
description: Diagnose a failure or performance anomaly. Read-only — this skill NEVER edits source, tests, or config. Outputs a root-cause report.
argument-hint: "<symptom: failing test, panic, perf regression>"
user-invocable: true
---

# molrs-debug

Read `CLAUDE.md` for molrs conventions. This skill is **diagnose-only**.
It exists so you stop and think before leaping to `/molrs-fix` or
`/molrs-refactor`. If you find yourself wanting to edit code, exit this
skill and run the appropriate write-skill instead.

## Procedure

1. **Classify the symptom** — one of:
   - `test-failure` — `cargo test` failing
   - `panic` — unwrap / expect / overflow in production path
   - `perf-regression` — benchmark slower than baseline
   - `numerical` — gradient mismatch, energy not conserved, wrong result
   - `ffi` — cross-boundary error (CXX, Python, WASM)
2. **Reproduce locally** — minimal command line. If you cannot reproduce,
   say so and stop; do not speculate.
3. **Bisect** — `git log` the implicated paths; if the symptom is
   regression-flavored, run `git bisect` on the smallest possible
   command (`cargo test -p <crate> <test> --release` — not the full
   suite). Report the first-bad commit.
4. **Root cause** — state the cause in one sentence and cite
   `file:line`. Distinguish cause from trigger (`cause: missing Sync
   bound on X; trigger: feature=rayon enabling shared access`).
5. **Suggest** — the right write-skill to apply next:
   - minimal logic bug → `/molrs-fix`
   - wrong boundary / module placement → `/molrs-refactor`
   - wrong design / new axis needed → `/molrs-spec` followed by
     `/molrs-impl`

## Rules

- **Never edit files.** Not source, not tests, not Cargo.toml, not
  docs. This skill's Write/Edit capabilities are not granted — the
  procedure enforces diagnose-only by delegation.
- State uncertainty explicitly. "I don't know why X happens" beats a
  confident wrong guess.

## Output

```
SYMPTOM:     <one line>
REPRO:       <command>
FIRST BAD:   <commit SHA or "not a regression">
ROOT CAUSE:  <one sentence> (<file>:<line>)
NEXT:        /molrs-fix | /molrs-refactor | /molrs-spec
```
