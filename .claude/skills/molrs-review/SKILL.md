---
name: molrs-review
description: Aggregate multi-axis code review. Fans out to architect, optimizer, documenter, scientist, ffi-safety agents in parallel and renders a single severity report. Does not edit code.
argument-hint: "[path or module — defaults to `git diff --name-only HEAD`]"
user-invocable: true
---

# molrs-review

Read `CLAUDE.md` for molrs conventions. This skill orchestrates a review;
rules for each axis live in the paired reference skills, which the
corresponding agent loads. Do NOT duplicate rule text here.

## Procedure

1. **Scope** — `$ARGUMENTS` if given, else `git diff --name-only HEAD`.
   Drop files outside the workspace (generated artifacts, `target/`,
   `pkg/`).

2. **Dispatch in parallel** — spawn all applicable agents in a single
   message. Conditional axes skip silently if no matching files:

   | Agent | Paired skill | When to invoke |
   |---|---|---|
   | `molrs-architect` | `molrs-arch` | Always |
   | `molrs-tester` | `molrs-test` | Always |
   | `molrs-optimizer` | `molrs-perf` | Hot-path code (`molrs-ff/potential/**`, `molrs-core/neighbors/**`, `molrs-pack/objective.rs`, `molrs-pack/packer.rs`) |
   | `molrs-documenter` | `molrs-doc` | Any `pub` item added or changed |
   | `molrs-scientist` | `molrs-science` | Physics touched (potentials, integrators, constraints, stereo) |
   | `molrs-ffi-safety` | `molrs-ffi` | Any change under `molrs-cxxapi/`, `molrs-python/src/`, `molrs-wasm/src/`, `molrs-capi/` |
   | `molrs-docs-engineer` | `molrs-docs` | PyO3/wasm-bindgen bindings added or renamed, or any `docs/**` change |

   Inline checks (no dedicated agent):

   - **Code quality** — functions < 50 lines, files < 800 lines, nesting
     ≤ 4 levels, `cargo clippy` clean, `cargo fmt` compliant.
   - **Immutability** — input Frame/Block not mutated; clone before
     modification; owned vs borrowed semantics correct.

3. **Aggregate** — collect agent outputs (each agent emits
   `[SEVERITY] file:line — message` lines). Render the severity table
   below.

## Severity

- **CRITICAL** — safety / architecture violation, wrong physics, FFI
  panic risk. Block merge.
- **HIGH** — missing tests, performance regression, wrong units.
- **MEDIUM** — style, documentation gaps.
- **LOW** — nice to have.

## Output

```
CODE REVIEW: <path>

ARCHITECTURE:   <findings>
TESTING:        <findings>
PERFORMANCE:    <findings | skipped: no hot-path files>
DOCUMENTATION:  <findings | skipped: no pub API touched>
SCIENCE:        <findings | skipped: no physics touched>
FFI SAFETY:     <findings | skipped: no FFI surface touched>
DOCS SYSTEM:    <findings | skipped: no bindings/docs touched>
CODE QUALITY:   <inline checklist>
IMMUTABILITY:   <inline checklist>

SUMMARY: N CRITICAL, N HIGH, N MEDIUM, N LOW
VERDICT: APPROVE | REQUEST CHANGES | BLOCK
```

End with a one-line recap: `reviewed <N> files, <axes-run> agents, <verdict>`.
