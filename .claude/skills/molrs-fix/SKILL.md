---
name: molrs-fix
description: Minimal-diff bug fix — write a regression test first, then the smallest change that makes it pass. Writes tests and one or more source files.
argument-hint: "<bug description or issue #>"
user-invocable: true
---

# molrs-fix

Read `CLAUDE.md` for molrs conventions. This skill is distinct from
`/molrs-impl`: it is for **bug fixes**, not new features. No architecture
phase, no docs phase, no multi-agent fan-out unless the root cause crosses
an axis.

## Procedure

1. **Reproduce** — turn the bug report into a failing unit or integration
   test in the crate that owns the faulty code. Use the patterns from the
   `molrs-test` reference skill (gradient, round-trip, PBC, etc.) — the bug
   usually has a canonical shape. Run the test and confirm RED.
2. **Diagnose narrowly** — read only the files implicated by the stack
   trace or by `cargo test` output. Do NOT broaden scope. If the root cause
   is in a different crate than you expected, re-run step 1 against that
   crate.
3. **Minimal fix** — change the smallest number of lines that turns the
   test GREEN. Resist the urge to refactor surrounding code; if you see
   something that wants refactoring, capture it with `/molrs-note` for a
   later `/molrs-refactor` run.
4. **Delegate if the axis is specialized**:
   - Wrong physics / wrong sign → spawn `molrs-scientist` agent to verify
     the fix against a reference before committing.
   - Hot-path regression → spawn `molrs-optimizer`.
   - FFI / extern panic → spawn `molrs-ffi-safety`.
5. **Verify** — `cargo test -p <crate>` GREEN; `cargo clippy -- -D
   warnings`; `cargo fmt --all`.

## Rules

- One bug, one commit. No incidental cleanup. No new public API.
- The regression test must fail before the fix and pass after. If you
  cannot make it fail, the bug is not yet reproduced — return to step 1.
- Never modify the regression test to make it pass.

## Output

One line: `fixed <issue>: <crate>/<file>:<line>, +<N> -<M> LOC, 1 test
added`.
