---
name: molrs-test
description: Testing standards for molrs Rust workspace — numerical gradient verification, reference comparison, PBC edge cases, round-trip I/O. Reference document only; no procedural workflow.
---

Reference standard for molrs testing. The `molrs-tester` agent applies these rules; this file defines them.

## Test Organization

```
molrs-<crate>/tests/         # Integration tests
molrs-<crate>/src/**/tests.rs  # Unit tests (inline #[cfg(test)] modules)
molrs-<crate>/benches/       # Criterion benchmarks
```

Each workspace crate (`molrs-core`, `molrs-io`, `molrs-compute`, `molrs-smiles`, `molrs-ff`, `molrs-embed`, `molrs-pack`, `molrs-cxxapi`) follows the same pattern.

### Running Tests

```bash
cargo test --all-features                    # all tests
cargo test -p molrs-core                     # single crate
cargo test -p molrs-core test_name           # single test
cargo test --features slow-tests             # expensive tests
cargo test -p molrs-pack --test examples_batch -- --ignored  # batch validation
```

### Test Data

Real test data must be fetched once:

```bash
bash scripts/fetch-test-data.sh   # clones to molrs-core/target/tests-data/
```

**IO testing rule (MANDATORY)**: format readers/writers MUST be tested against every real file in `tests-data/<format>/` — never against synthetic strings. See `CLAUDE.md` "IO Testing Rules" for the full policy.

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

Constraints accumulate TRUE gradient (`∂V/∂x`) with `+=`; numerical gradient must match.

### 6. Rotation Convention (multi-atom MUST)

LEFT multiplication: `R_new = δR * R_old`. Single-atom tests CANNOT catch LEFT/RIGHT mult bugs (rotation gradient is zero) — use ≥ 3 atoms with non-collinear positions.

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

## Coverage Target

≥ 80% per crate. Mark expensive tests with `#[cfg(feature = "slow-tests")]`.

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
