---
name: molrs-refactor
description: Restructure code without changing observable behavior. Enforces the extraction discipline from `molrs-perf` on hot-path monoliths. Writes source files; tests and benchmarks must already exist.
argument-hint: "<what to restructure and why>"
user-invocable: true
---

# molrs-refactor

Read `CLAUDE.md` for molrs conventions. This skill is distinct from
`/molrs-impl` (new feature) and `/molrs-fix` (bug). It restructures
existing, passing code. No new public API; no behavior change.

## Procedure

1. **Pre-flight** — confirm the target area has:
   - Green tests (`cargo test -p <crate>`)
   - A pinned benchmark if the target is a hot path (see `molrs-perf`
     skill § Hot Path Hierarchy)
   If either is missing, STOP. Ask the user to land tests or a bench
   first — refactoring without them is unsafe at this layer.
2. **Architect review (before)** — spawn `molrs-architect` agent with
   the refactor goal. Agent returns a module impact map and flags any
   rule breaks in the current layout. Use the map to plan the commits.
3. **Hot-path extraction discipline** — if touching any of:
   `molrs-pack/src/{objective,packer}.rs`, `molrs-ff/src/potential/**`,
   `molrs-core/src/neighbors/**`, or any file `molrs-perf` lists as a
   hot path, follow the extraction protocol in that skill § "Benchmarking
   during refactors" — one extraction per commit, with `#[cfg(bench)]
   #[inline(never)] fn F_sentinel`, extracted-fn microbench, caller
   microbench, and gates (+1% extracted, +2% caller). Not optional.
4. **Execute** — one commit per logical restructure. Tests stay GREEN
   throughout; no commit may land RED, even transiently. Use
   `cargo test -p <crate> && cargo clippy -- -D warnings` between
   commits.
5. **Architect review (after)** — spawn `molrs-architect` again against
   the final state. For hot-path work, spawn `molrs-optimizer` to
   confirm sentinels landed and gates were met.

## Rules

- No new public API. If the refactor suggests one, split that out into a
  `/molrs-spec` + `/molrs-impl` cycle after this one.
- No deprecated-shim parallel modules (`foo_v2.rs` alongside `foo.rs`).
  molrs refactors are in-place (see memory: "Refactors — no
  backwards-compat shims"). Transient non-building state during a multi-
  commit refactor is acceptable; shipping both old and new is not.
- Never delete a sentinel in the same commit it was introduced. Sentinels
  live for one follow-up refactor cycle, then they're swept.

## Output

One line: `refactored <module>: N commits, M extractions, all benches
within gate`.
