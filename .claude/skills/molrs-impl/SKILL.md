---
name: molrs-impl
description: Orchestrate multi-agent feature development for molrs. Takes a feature description or spec and coordinates architecture review, TDD, implementation, code review, performance review, and FFI safety.
---

You are the **molrs implementation orchestrator**. Given a feature description or spec, you coordinate the full development lifecycle using specialized agents.

**Execution discipline**: Before writing any code, enter **Plan Mode** to lay out the full plan, then create **Tasks** for each phase below. Update task status as work progresses (`in_progress` → `completed`). This enforces a structured, auditable workflow — the agent must not skip phases or jump ahead without completing prior tasks.

## Trigger

`/molrs-impl <feature description or spec path>`

## Workflow

### Phase 0: Understand & Plan

1. **Parse the input**: Determine if the input is a natural language description or a path to an existing spec.
2. **If no spec exists**: Suggest running `/molrs-spec` first to generate one. Offer to proceed with informal planning if the user prefers.
3. **Read the spec** and identify:
   - Which crates are affected (`molrs-core`, `molrs-md`, `molrs-pack`, `molrs-ffi`, `molrs-wasm`)
   - Which traits are extended or created
   - Which data structures are modified
   - Whether FFI/WASM bindings need updates
   - Whether CUDA kernels are involved

### Phase 1: Architecture Review

Launch the **architect** agent (or `everything-claude-code:planner`) to:
- Validate the feature fits the existing trait-based architecture
- Check dependency flow: `molrs-core ← molrs-ffi ← molrs-wasm`, `molrs-core ← molrs-md`, `molrs-core ← molrs-pack`
- Identify which module(s) the feature belongs in
- Confirm the feature doesn't violate:
  - Handle-based FFI (no raw pointers crossing boundaries)
  - Float precision abstraction (use `F` alias, not `f32`/`f64`)
  - ndarray conventions for coordinates
  - Immutability principles (return new objects, don't mutate)
- Output: Implementation plan with file list, trait signatures, and data flow

### Phase 2: Interface Design (TDD - RED)

Launch the **tdd-guide** agent to:
1. Define the public API (trait signatures, struct definitions, function signatures)
2. Write test cases FIRST:
   - Unit tests for each function/method
   - Integration tests for trait implementations
   - Numerical accuracy tests (compare against reference implementations)
   - Edge cases: empty inputs, single-atom systems, zero vectors, PBC wrapping
3. For potential/force kernels: include numerical gradient verification tests
4. For packing changes: include Packmol reference comparison tests
5. Run tests — they should all FAIL (RED)

### Phase 3: Implementation (TDD - GREEN)

Write minimal code to pass all tests:
- Follow file organization: 200-400 lines per file, 800 max
- Use the `F` type alias for all floats
- Implement `Send + Sync` for any trait object that crosses thread boundaries
- For `Potential` implementations: `eval(&self, coords: &[F]) -> (F, Vec<F>)` signature
- For `Fix` implementations: declare GPU tier (`Kernel`/`Async`/`Sync`)
- For `Constraint` implementations: accumulate TRUE gradient with `+=`
- Run tests — they should all PASS (GREEN)

### Phase 4: Review & Refactor (TDD - IMPROVE)

Launch agents in parallel:

1. **code-reviewer** agent: Check code quality, readability, Rust idioms
2. **molrs-perf** skill: Check for performance issues (see `.claude/skills/molrs-perf/`)
3. **molrs-ffi** skill (if FFI affected): Verify handle safety (see `.claude/skills/molrs-ffi/`)

Address findings, then:
- Run `cargo clippy -- -D warnings`
- Run `cargo fmt --all`
- Verify test coverage >= 80%

### Phase 5: Documentation

If public API was added/changed:
- Add doc comments (`///`) on all public items
- Include usage examples in doc comments
- Update crate-level docs if a new module was added

### Phase 6: Integration Verification

```bash
cargo build --all-features
cargo test --all-features
cargo clippy -- -D warnings
cargo bench -p <affected-crate>    # if performance-sensitive
```

## Agent Dispatch Table

| Phase | Agent/Skill | Subagent Type | When |
|---|---|---|---|
| 1 | architect | `everything-claude-code:planner` | Always |
| 2 | tdd-guide | `everything-claude-code:tdd-guide` | Always |
| 3 | (self) | — | Always |
| 4a | code-reviewer | `everything-claude-code:code-reviewer` | Always |
| 4b | molrs-perf | — | Performance-sensitive code |
| 4c | molrs-ffi | — | FFI boundary changes |
| 5 | (self) | — | Public API changes |
| 6 | build-error-resolver | `everything-claude-code:build-error-resolver` | If build fails |

## Crate-Specific Checklists

### molrs-core changes
- [ ] Uses `F` type alias (not raw f32/f64)
- [ ] ndarray for coordinates (not Vec<Vec<F>>)
- [ ] Trait objects are `Send + Sync`
- [ ] No `Cell<f64>` in Sync contexts (use AtomicU64)
- [ ] Kernel registered in `KernelRegistry` (if new potential)

### molrs-md changes
- [ ] Fix declares GPU tier
- [ ] Fix implements all relevant stage methods
- [ ] Dump is read-only (no state mutation)
- [ ] MDState properly initialized

### molrs-pack changes
- [ ] Constraints accumulate TRUE gradient with `+=`
- [ ] Rotation uses LEFT multiplication
- [ ] Checked against Packmol source (cite file:line)
- [ ] Numerical gradient test included

### molrs-ffi changes
- [ ] Handle-based (no raw pointers)
- [ ] Version counter incremented on mutation
- [ ] Invalid handle returns error (not panic)
- [ ] `#[no_mangle]` and `extern "C"` on FFI functions

### molrs-wasm changes
- [ ] Uses `wasm_bindgen` attributes
- [ ] Serde for complex type serialization
- [ ] No panics in WASM-exported functions (use Result)
