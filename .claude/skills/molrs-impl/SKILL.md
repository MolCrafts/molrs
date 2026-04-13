---
name: molrs-impl
description: Orchestrate multi-agent feature development for molrs. Entry point for any new feature — coordinates spec → architecture → TDD → implementation → review → docs.
---

You are the **molrs implementation orchestrator**. This is the single entry point for new feature work. You coordinate the full lifecycle using in-repo `molrs-*` agents and skills.

**Execution discipline**: Before writing any code, enter **Plan Mode** to lay out the full plan, then create **Tasks** for each phase below. Update task status as work progresses (`in_progress` → `completed`). Don't skip phases or jump ahead.

## Trigger

`/molrs-impl <feature description or spec path>`

## Phase 0 — Spec

If the input is a natural language description (not a path to an existing spec), invoke the `molrs-spec` skill internally to produce one. Do NOT bounce the user out to `/molrs-spec` — that creates a circular handoff.

If the input is already a spec path, read it.

Identify from the spec:

- Which crate(s) own the feature (8 active members):
  `molrs-core`, `molrs-io`, `molrs-compute`, `molrs-smiles`, `molrs-ff`, `molrs-gen3d`, `molrs-pack`, `molrs-cxxapi`
- Which traits are extended or created
- Which data structures are modified
- Whether FFI bindings (`molrs-cxxapi`) need updates

## Phase 1 — Architecture Review

Spawn the **molrs-architect** agent (subagent_type: `molrs-architect`) to validate the design against `molrs-arch` standards: crate dependency flow, module ownership, trait conformance, FFI safety, naming.

Output: implementation plan with file list, trait signatures, data flow.

## Phase 2 — Test Design (TDD — RED)

Spawn the **molrs-tester** agent (subagent_type: `molrs-tester`) to:

1. Define the public API (trait signatures, struct fields, function signatures).
2. Write tests FIRST per `molrs-test` standards:
   - Unit tests per function/method
   - Integration tests for trait implementations
   - Numerical gradient verification (for potentials/constraints)
   - PBC edge cases, round-trip I/O
   - Edge cases: empty, single atom, collinear, zero distance
3. For new physics: also spawn **molrs-scientist** to verify equation + units against literature.
4. Run tests — they should all FAIL (RED).

## Phase 3 — Implementation (TDD — GREEN)

Write minimal code to pass all tests:

- Files 200–400 lines (800 max)
- `F = f64` alias for floats
- `Send + Sync` for shared trait objects
- `Potential` signature: `eval(&self, coords: &[F]) -> (F, Vec<F>)`
- `Constraint`: accumulate TRUE gradient with `+=`
- Run tests — they should all PASS (GREEN).

## Phase 4 — Review & Refactor (TDD — IMPROVE)

Spawn agents in parallel:

| Agent | When |
|---|---|
| `molrs-architect` | Always — re-verify dependency flow & boundary rules |
| `molrs-optimizer` | Performance-sensitive code (potentials, neighbors, hot loops) |
| `molrs-documenter` | Public API additions/changes |
| `molrs-scientist` | Anything touching physics (force fields, integrators, constraints) |

For FFI changes (`molrs-cxxapi`): consult `molrs-ffi` skill checklist directly.

Address findings, then:

```bash
cargo clippy -- -D warnings
cargo fmt --all
```

Verify coverage ≥ 80%.

## Phase 5 — Documentation

If public API was added or changed, apply `molrs-doc` standards (docstring tiers, units, references). The `molrs-documenter` agent (already invoked Phase 4) will have flagged gaps.

## Phase 6 — Integration Verification

```bash
cargo build --all-features
cargo test --all-features
cargo clippy -- -D warnings
cargo bench -p <affected-crate>     # if performance-sensitive
```

## Agent Dispatch Table

| Phase | Agent / Skill | Subagent type | When |
|---|---|---|---|
| 0 | `molrs-spec` (skill) | — | If no spec exists |
| 1 | `molrs-architect` | `molrs-architect` | Always |
| 2 | `molrs-tester` | `molrs-tester` | Always |
| 2b | `molrs-scientist` | `molrs-scientist` | New physics |
| 3 | (self) | — | Always |
| 4a | `molrs-architect` | `molrs-architect` | Always |
| 4b | `molrs-optimizer` | `molrs-optimizer` | Performance-sensitive |
| 4c | `molrs-documenter` | `molrs-documenter` | Public API touched |
| 4d | `molrs-scientist` | `molrs-scientist` | Physics touched |
| 5 | `molrs-doc` (skill) | — | Public API touched |
| 6 | (self) | — | Always |

## Crate-Specific Checklists

### molrs-core changes
- [ ] Uses `F` type alias (not raw `f64`)
- [ ] ndarray for coordinates (not `Vec<Vec<F>>`)
- [ ] Trait objects are `Send + Sync`
- [ ] No `Cell<f64>` in Sync contexts (use `AtomicU64`)

### molrs-ff changes
- [ ] New kernel registered in `KernelRegistry`
- [ ] Numerical gradient test included
- [ ] Equation + reference cited in rustdoc

### molrs-pack changes
- [ ] Constraints accumulate TRUE gradient with `+=`
- [ ] Rotation uses LEFT multiplication
- [ ] Checked against Packmol source (cite `file:line`)

### molrs-io changes
- [ ] Tested against all real files in `tests-data/<format>/` (no synthetic strings)
- [ ] Round-trip test: read → write → read → equality

### molrs-cxxapi changes
- [ ] CXX bridge uses `#[cxx::bridge]`, no raw pointers
- [ ] Zero-copy via `FrameView` where possible
- [ ] Owned `Frame` only when persisting (Zarr)
- [ ] No panics in bridge functions
