---
name: molrs-test
description: Testing guidance for molrs Rust workspace. Covers numerical gradient verification, reference comparison tests, PBC edge cases, and molecular simulation-specific test patterns.
---

You are a **scientific software testing specialist** for the molrs molecular simulation workspace.

## Trigger

Use when writing tests for molrs code, reviewing test coverage, or debugging test failures.

## Test Organization

```
molrs-core/tests/          # Integration tests
molrs-core/src/**/tests.rs # Unit tests (inline #[cfg(test)] modules)
molrs-pack/tests/          # Integration + batch validation tests
molrs-md/tests/            # MD integration tests
```

### Running Tests

```bash
cargo test --all-features                    # all tests
cargo test -p molrs-core                     # single crate
cargo test -p molrs-core test_name           # single test
cargo test --features slow-tests             # expensive tests
cargo test -p molrs-pack --test examples_batch -- --ignored  # batch validation
```

### Test Data

Test data must be fetched once:
```bash
bash scripts/fetch-test-data.sh   # clones to molrs-core/target/tests-data/
```

## Molecular Simulation Test Patterns

### 1. Numerical Gradient Verification

Every potential kernel and constraint MUST have a numerical gradient test:

```rust
#[test]
fn test_gradient_numerical() {
    let kernel = MyKernel::new(params);
    let coords: Vec<F> = vec![/* test coordinates */];
    let (energy, analytical_grad) = kernel.eval(&coords);

    let h: F = 1e-5;
    for i in 0..coords.len() {
        let mut coords_plus = coords.clone();
        let mut coords_minus = coords.clone();
        coords_plus[i] += h;
        coords_minus[i] -= h;
        let (e_plus, _) = kernel.eval(&coords_plus);
        let (e_minus, _) = kernel.eval(&coords_minus);
        let numerical = (e_plus - e_minus) / (2.0 * h);
        assert!(
            (analytical_grad[i] - numerical).abs() < 1e-4,
            "gradient mismatch at index {}: analytical={}, numerical={}",
            i, analytical_grad[i], numerical
        );
    }
}
```

**Step size**: `h = 1e-5` for f32, `h = 1e-7` for f64.
**Tolerance**: `1e-4` for f32, `1e-6` for f64.

### 2. Newton's Third Law Test

For pair potentials, forces must be equal and opposite:

```rust
#[test]
fn test_newtons_third_law() {
    let kernel = PairKernel::new(params);
    let coords = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
    let (_, forces) = kernel.eval(&coords);
    assert!((forces[0] + forces[3]).abs() < 1e-6);
    assert!((forces[1] + forces[4]).abs() < 1e-6);
    assert!((forces[2] + forces[5]).abs() < 1e-6);
}
```

### 3. Energy Conservation Test (MD)

For symplectic integrators (NVE), total energy must be conserved:

```rust
#[test]
fn test_energy_conservation_nve() {
    let engine = build_nve_engine(/* params */);
    let state = engine.init(&frame);
    let mut energies = Vec::new();
    for _ in 0..1000 {
        let state = engine.run(1, state);
        energies.push(state.kinetic_energy + state.potential_energy);
    }
    let drift = (energies.last().unwrap() - energies.first().unwrap()).abs();
    let mean_energy = energies.iter().sum::<F>() / energies.len() as F;
    assert!(drift / mean_energy < 1e-4, "energy drift too large: {}", drift);
}
```

### 4. PBC Edge Cases

Always test periodic boundary conditions:

```rust
#[test]
fn test_minimum_image_wrapping() {
    let simbox = SimBox::cubic(10.0);
    let r1 = array![0.5, 0.0, 0.0];
    let r2 = array![9.5, 0.0, 0.0];
    let dist = simbox.calc_distance_impl(&r1, &r2);
    assert!((dist - 1.0).abs() < 1e-6); // wrapped distance = 1.0, not 9.0
}

#[test]
fn test_non_periodic_axis() {
    let simbox = SimBox::new(/* ... */, pbc: [true, true, false]);
    let r1 = array![0.0, 0.0, 0.5];
    let r2 = array![0.0, 0.0, 9.5];
    let dist = simbox.calc_distance_impl(&r1, &r2);
    assert!((dist - 9.0).abs() < 1e-6); // NOT wrapped
}
```

### 5. Constraint Gradient Sign Convention

Constraints accumulate TRUE gradient (dV/dx) with `+=`:

```rust
#[test]
fn test_constraint_gradient_sign() {
    let constraint = InsideBoxConstraint::new(/* bounds */);
    let coords = vec![/* outside box */];
    let (violation, gradient) = constraint.eval(&coords);
    assert!(violation > 0.0, "should report violation when outside");

    // Verify with numerical gradient
    let h = 1e-5;
    for i in 0..coords.len() {
        let mut cp = coords.clone();
        let mut cm = coords.clone();
        cp[i] += h;
        cm[i] -= h;
        let (vp, _) = constraint.eval(&cp);
        let (vm, _) = constraint.eval(&cm);
        let num_grad = (vp - vm) / (2.0 * h);
        assert!((gradient[i] - num_grad).abs() < 1e-4);
    }
}
```

### 6. Rotation Convention Test

Must use LEFT multiplication and test with multi-atom systems:

```rust
#[test]
fn test_rotation_gradient_multi_atom() {
    // CRITICAL: Single-atom tests won't catch LEFT vs RIGHT mult bugs
    // because rotation gradient is zero for atoms at the origin
    let positions = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    // ... test that gradient descent actually reduces the objective
}
```

### 7. Round-Trip I/O Test

```rust
#[test]
fn test_pdb_round_trip() {
    let frame = read_pdb("test_data/input.pdb").unwrap();
    let mut buf = Vec::new();
    write_pdb(&frame, &mut buf).unwrap();
    let frame2 = read_pdb_from_bytes(&buf).unwrap();
    assert_eq!(
        frame.get("atoms").unwrap().nrows(),
        frame2.get("atoms").unwrap().nrows()
    );
}
```

### 8. Empty / Degenerate Input Tests

Always test edge cases:

```rust
#[test] fn test_empty_frame() { /* 0 atoms */ }
#[test] fn test_single_atom() { /* no pairs, no bonds */ }
#[test] fn test_collinear_atoms() { /* angle = 0 or pi */ }
#[test] fn test_zero_distance() { /* overlapping atoms */ }

#[test]
#[cfg(feature = "slow-tests")]
fn test_huge_system() { /* 10K+ atoms */ }
```

## Test Quality Checklist

- [ ] Every potential kernel has numerical gradient test
- [ ] Every constraint has gradient sign convention test
- [ ] Pair potentials have Newton's third law test
- [ ] MD integrators have energy conservation test
- [ ] I/O formats have round-trip test
- [ ] PBC-sensitive code has wrapping edge case test
- [ ] Multi-atom rotation tests (single-atom won't catch LEFT/RIGHT bugs)
- [ ] Edge cases: empty, single atom, collinear, zero distance
- [ ] Expensive tests gated behind `#[cfg(feature = "slow-tests")]`
- [ ] Test data fetched via `scripts/fetch-test-data.sh`
