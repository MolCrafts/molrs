---
name: molrs-review
description: Comprehensive code review aggregating architecture, performance, documentation, and Rust safety checks. Use after writing code or during PR review.
argument-hint: "[path or module]"
user-invocable: true
---

Review code for: $ARGUMENTS

If no path given, review all files modified in `git diff --name-only HEAD`.

**Invoke all dimensions in parallel:**

1. **Architecture** → invoke `/molrs-arch` on $ARGUMENTS
2. **Performance** → invoke `/molrs-perf` on $ARGUMENTS
3. **Documentation** → invoke `/molrs-doc` on $ARGUMENTS
4. **Rust & FFI Safety**:
   - Float precision: all numeric code uses `F` alias, not raw f32/f64
   - Coordinate convention: flat `[x0,y0,z0, x1,y1,z1, ...]` format
   - Trait conformance: Potential/Fix/Dump implement required methods
   - Registration patterns: KernelRegistry used correctly
   - FFI handle safety: SlotMap handles, no raw pointers across boundary
   - WASM memory: proper free() calls, no use-after-free
   - Constraint gradients: TRUE gradient with `+=`, optimizer negates
   - Rotation: LEFT multiplication `R_new = δR * R_old`
   - `Cell<f64>` is NOT Sync — use AtomicU64 with to_bits/from_bits
5. **Code Quality** (inline):
   - Functions < 50 lines, files < 800 lines
   - No deep nesting (> 4 levels)
   - Descriptive naming (no single-letter except loop indices)
   - `cargo clippy` clean
   - `cargo fmt` compliant
6. **Immutability** (inline):
   - Input Frame/Block not mutated
   - Clone before modification
   - Owned vs borrowed semantics correct

**Severity levels**:
- CRITICAL — must fix (safety issues, architecture violations)
- HIGH — should fix (missing tests, performance issues)
- MEDIUM — fix when possible (style, documentation gaps)
- LOW — nice to have

**Output**: Merged report:
```
CODE REVIEW: <path>
ARCHITECTURE: ✅/❌ per check
PERFORMANCE: ✅/⚠️ per check
DOCUMENTATION: ✅/⚠️ per check
RUST & FFI SAFETY: ✅/❌ per check
CODE QUALITY: ✅/⚠️ per check
IMMUTABILITY: ✅/❌ per check
SUMMARY: N CRITICAL, N HIGH, N MEDIUM, N LOW
```
