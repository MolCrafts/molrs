---
name: molrs-review
description: Comprehensive code review aggregating architecture, performance, documentation, scientific correctness, and FFI safety. Use after writing code or during PR review.
argument-hint: "[path or module]"
user-invocable: true
---

Review code for: $ARGUMENTS

If no path given, review all files modified in `git diff --name-only HEAD`.

## Workflow

Spawn the relevant `molrs-*` agents in parallel. Each agent applies its corresponding skill's standards and reports findings. Do NOT duplicate the rule lists here — they live in the skills.

| Dimension | Agent | Skill (standards) | When to invoke |
|---|---|---|---|
| Architecture | `molrs-architect` | `molrs-arch` | Always |
| Performance | `molrs-optimizer` | `molrs-perf` | Hot-path code (potentials, neighbors, GENCAN inner loop) |
| Documentation | `molrs-documenter` | `molrs-doc` | Public API touched |
| Scientific correctness | `molrs-scientist` | `molrs-science` | Physics touched (potentials, integrators, constraints) |
| FFI safety | (inline review) | `molrs-ffi` | `molrs-cxxapi` touched |

For code-quality and immutability dimensions that do not have dedicated agents, apply inline:

- **Code quality**: functions < 50 lines, files < 800 lines, nesting ≤ 4 levels, descriptive naming, `cargo clippy` clean, `cargo fmt` compliant
- **Immutability**: input Frame/Block not mutated; clone before modification; owned vs borrowed semantics correct

## Severity

- **CRITICAL** — safety / architecture violation. Block merge.
- **HIGH** — missing tests, performance regression, wrong physics.
- **MEDIUM** — style, documentation gaps.
- **LOW** — nice to have.

## Output Format

```
CODE REVIEW: <path>

ARCHITECTURE:    <findings from molrs-architect>
PERFORMANCE:     <findings from molrs-optimizer>
DOCUMENTATION:   <findings from molrs-documenter>
SCIENCE:         <findings from molrs-scientist>
FFI SAFETY:      <inline review against molrs-ffi skill, if applicable>
CODE QUALITY:    <inline checklist results>
IMMUTABILITY:    <inline checklist results>

SUMMARY: N CRITICAL, N HIGH, N MEDIUM, N LOW
```
