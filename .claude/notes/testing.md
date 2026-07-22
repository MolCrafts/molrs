# Testing Standards

Project standard for molrs testing. Applied by the `mol:tester` agent,
`/mol:test`, and `/mol:impl`.

## Test Organization

Merged crate `molcrafts-molrs` (dir `molrs/`):

```
molrs/tests/                 # Integration / data-driven / architecture gates
  <module>/…                 # mirrors molrs/src/<module>/ (io, ff, compute, …)
  architecture_gate.rs       # structural anti-pattern gates (no silent skip, …)
molrs/src/**                 # Unit tests: inline #[cfg(test)] modules only
molrs/benches/               # Criterion benchmarks
```

Workspace members that still ship their own trees:

- `molrs-cxxapi/` — FFI handle-lifecycle tests (see `.claude/notes/ffi.md`)
- Binder workspaces (`molrs-python`, `molrs-wasm`, …) are **separate**; do not
  assume they share this suite.

### Running Tests

```bash
# Default / CI gate — unit only (src/ #[cfg(test)])
cargo test -p molcrafts-molrs --lib --features full

# Integration binaries under molrs/tests/ (IO, ff stages, gates) — full.yml / manual
bash scripts/fetch-test-data.sh
cargo test -p molcrafts-molrs --tests --features "full filesystem"

cargo test -p molcrafts-molrs test_name --features full   # single test
```

### Test Data

Real test data must be fetched once:

```bash
bash scripts/fetch-test-data.sh   # clones to <root>/tests-data/ (binding-neutral)
```

**IO testing rule (MANDATORY)**: format readers/writers MUST be tested against
every real file in `tests-data/<format>/` — never against synthetic strings.
See `CLAUDE.md` "IO Testing Rules" for the full policy. Synthetic
`let content = "..."; read_from_str(content)` is permitted ONLY for
malformed-input edge cases, never for happy paths.

Helpers (io test target only):

| Helper | Role |
|---|---|
| `common::tests_data_dir()` | locate `tests-data/` (or `$MOLRS_TESTS_DATA`); panics if absent |
| `common::format_files("<fmt>")` | every file in `tests-data/<fmt>/`; panics if empty |
| `common::require_fixture("fmt/name.ext")` | one **required** file; panics if missing |

## Iron laws (tester)

### 1. No vacuous green

A test that asserts nothing is not coverage.

| Forbidden | Allowed |
|---|---|
| `if !path.exists() { return; }` / `continue;` | `common::require_fixture(…)` or hard `assert!(path.is_file())` |
| `println!("skipping…"); return;` | delete the test, vendor the input, or honest `#[ignore = "reason"]` |
| empty loop over a missing/empty fixture dir | panic if the corpus is required; `assert!(!paths.is_empty())` |

Gate: `architecture_gate::no_test_returns_green_when_its_input_is_absent`.

### 2. No hand-written fixture subsets

If a partition can be **computed** from the fixtures (charge sum, atom types,
directory scan), it must not be a hand-written name list. Subsets that omit
the only molecule that can fail are how 150 kcal/mol holes ship green.

- Completeness lists (`REQUIRED_FIXTURES`) are OK only to prove nothing was
  **deleted** — parity loops must still iterate the scan.
- If a subset truly cannot be computed, state the reason **in the test**.
  "Not yet implemented" is a reason to **fail**, not to exclude.

Gate: `architecture_gate::no_test_asserts_on_a_subset_of_its_fixtures`.

### 3. Hard-coded goldens for oracles

External oracles (RDKit, antechamber, …) are frozen offline into fixtures /
JSON. Tests must not spawn live third-party tools for pass/fail (generator
compile checks are the exception and are local).

### 4. Layout & granularity

- Unit tests: pure logic / edge / error paths next to production (`#[cfg(test)]`).
- Integration / format / multi-stage: `molrs/tests/`, path mirrors `src/`.
- One concern per test function. Full product stories are not unit tests.
- No wall-clock nondeterminism; fixed seeds where randomness is required.

## Molecular Simulation Test Patterns

### 1. Numerical Gradient Verification

Every potential kernel and constraint MUST have a numerical gradient test:

```rust
#[test]
fn test_gradient_numerical() {
    let kernel = MyKernel::new(params);
    let coords: Vec<F> = vec![/* test coordinates */];
    let (_, analytical) = kernel.eval(&coords);

    let h: F = 1e-7;             // F = f64
    let tol: F = 1e-6;
    for i in 0..coords.len() {
        let mut cp = coords.clone(); cp[i] += h;
        let mut cm = coords.clone(); cm[i] -= h;
        let (ep, _) = kernel.eval(&cp);
        let (em, _) = kernel.eval(&cm);
        let numerical = (ep - em) / (2.0 * h);
        assert!(
            (analytical[i] - numerical).abs() < tol,
            "gradient mismatch at {i}: analytical={}, numerical={}",
            analytical[i], numerical
        );
    }
}
```

`F = f64` always — use `h = 1e-7`, `tol = 1e-6`.

### 2. Newton's Third Law

Pair potentials: forces equal and opposite.

```rust
let coords = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
let (_, f) = kernel.eval(&coords);
assert!((f[0] + f[3]).abs() < 1e-6);
assert!((f[1] + f[4]).abs() < 1e-6);
assert!((f[2] + f[5]).abs() < 1e-6);
```

### 3. Energy Conservation (NVE)

Symplectic integrators: total energy conserved.

```rust
let drift = (energies.last().unwrap() - energies.first().unwrap()).abs();
let mean = energies.iter().sum::<F>() / energies.len() as F;
assert!(drift / mean < 1e-4, "energy drift too large: {drift}");
```

### 4. PBC Edge Cases

Always test minimum-image wrapping AND non-periodic axes.

```rust
let simbox = SimBox::cubic(10.0);
let r1 = array![0.5, 0.0, 0.0];
let r2 = array![9.5, 0.0, 0.0];
assert!((simbox.calc_distance_impl(&r1, &r2) - 1.0).abs() < 1e-6);
```

### 5. Constraint Gradient Sign

Constraints accumulate TRUE gradient (`∂V/∂x`) with `+=`; numerical gradient
must match. (Packing constraints now live in the molpack repo.)

### 6. Rotation Convention (multi-atom MUST)

LEFT multiplication: `R_new = δR * R_old`. Single-atom tests CANNOT catch
LEFT/RIGHT mult bugs (rotation gradient is zero) — use ≥ 3 atoms with
non-collinear positions.

### 7. Round-Trip I/O

```rust
let frame = read_pdb("test_data/input.pdb").unwrap();
let mut buf = Vec::new();
write_pdb(&frame, &mut buf).unwrap();
let frame2 = read_pdb_from_bytes(&buf).unwrap();
assert_eq!(frame.get("atoms").unwrap().nrows(), frame2.get("atoms").unwrap().nrows());
```

### 8. Edge Cases

```rust
#[test] fn test_empty_frame() { /* 0 atoms */ }
#[test] fn test_single_atom() { /* no pairs, no bonds */ }
#[test] fn test_collinear_atoms() { /* angle = 0 or π */ }
#[test] fn test_zero_distance() { /* overlapping atoms */ }

#[test]
#[cfg(feature = "slow-tests")]
fn test_huge_system() { /* 10K+ atoms */ }
```

Anything in `molrs-cxxapi` additionally needs FFI handle-lifecycle tests
(see `.claude/notes/ffi.md`).

## Coverage Target

≥ 80% per module surface. Mark expensive tests with `#[cfg(feature = "slow-tests")]`.

## Compliance Checklist

- [ ] Every potential kernel has numerical gradient test
- [ ] Every constraint has gradient sign convention test
- [ ] Pair potentials have Newton's 3rd law test
- [ ] MD integrators have energy conservation test
- [ ] I/O formats have round-trip test
- [ ] PBC-sensitive code has wrapping edge case test
- [ ] Rotation tests use multi-atom systems
- [ ] Edge cases: empty, single atom, collinear, zero distance
- [ ] Slow tests gated behind `#[cfg(feature = "slow-tests")]`
- [ ] IO tests iterate over `tests-data/<format>/*` (never synthetic)
- [ ] No silent soft-skip of missing fixtures
- [ ] Fixture partitions computed, not hand-named (except completeness lists)
