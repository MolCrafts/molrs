---
name: molrs-perf
description: Performance optimization guidance for molrs molecular simulation code. Covers hot loops, SIMD, parallelism, memory layout, neighbor lists, potential kernels, and CUDA integration.
---

You are a **high-performance computing specialist** for molecular simulation in Rust. You review and optimize molrs code for maximum throughput.

## Trigger

Use when writing or reviewing performance-sensitive code: potential kernels, neighbor lists, MD integrators, packing optimizers, or any code in the inner simulation loop.

## Performance-Critical Paths

### Hot Loop Hierarchy (most to least critical)

1. **Pair potential evaluation** -- O(N^2) or O(N*k) with neighbor lists. Called every MD step.
2. **Neighbor list build/update** -- O(N) with LinkCell, O(N^2) with BruteForce. Rebuilt periodically.
3. **Force accumulation** -- Summing forces from all potential terms.
4. **GENCAN inner loop** -- Objective + gradient evaluation in packing optimizer.
5. **MD integration** -- Position/velocity updates (O(N) per step, but frequent).

### Memory Layout Optimization

**Prefer Structure-of-Arrays (SoA) over Array-of-Structures (AoS)**:
```rust
// GOOD (SoA) -- cache-friendly for component-wise operations
let x: Array1<F> = ...;  // all x-coordinates contiguous
let y: Array1<F> = ...;
let z: Array1<F> = ...;

// ACCEPTABLE -- ndarray row-major layout
let coords: Array2<F> = Array2::zeros((n_atoms, 3));

// BAD -- pointer chasing, cache-hostile
let atoms: Vec<Atom> = ...;  // each Atom has x, y, z fields
```

The Zarr trajectory format uses SoA by design (per-component arrays).

### Flat Coordinate Vectors for Kernels

Potential kernels use flat `&[F]` (3N elements): `[x0,y0,z0, x1,y1,z1, ...]`. This enables:
- Direct SIMD vectorization
- Zero-copy FFI to CUDA
- Contiguous memory access

### Neighbor List Performance

`LinkCell` (default) is O(N) for build, O(N*k) for traversal (k = avg neighbors). Key optimizations:
- Cell size >= cutoff ensures only 27 neighboring cells checked
- `PairVisitor` trait enables zero-allocation traversal
- Rayon parallelism for build phase (feature-gated)

**Do NOT**:
- Rebuild neighbor list every step (use Verlet skin distance)
- Use `BruteForce` in production (O(N^2), testing only)
- Allocate per-pair during traversal (use `PairVisitor` callback)

## Optimization Techniques

### 1. Avoid Allocation in Hot Loops

```rust
// BAD: allocates Vec every call
fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
    let mut forces = vec![0.0; coords.len()];  // allocation!
    // ...
}

// BETTER: reuse buffer via parameter or self
fn eval_into(&self, coords: &[F], forces: &mut [F]) -> F {
    forces.fill(0.0);
    // ...
}
```

For the `Potential` trait, the signature requires returning `Vec<F>`. Minimize internal allocations.

### 2. SIMD-Friendly Patterns

```rust
// GOOD: simple loop, auto-vectorizable
for i in 0..n {
    forces[3*i]   += scale * dx;
    forces[3*i+1] += scale * dy;
    forces[3*i+2] += scale * dz;
}

// BAD: branch in inner loop prevents vectorization
for i in 0..n {
    if atoms[i].is_active {  // branch!
        forces[3*i] += scale * dx;
    }
}
```

### 3. Rayon Parallelism

Use rayon for embarrassingly parallel operations:
- Neighbor list cell processing
- Independent potential kernel evaluation
- Force accumulation with thread-local buffers

```rust
// Pattern: parallel reduce with thread-local accumulation
use rayon::prelude::*;

let (energy, forces) = pairs.par_chunks(chunk_size)
    .map(|chunk| {
        let mut local_e = 0.0;
        let mut local_f = vec![0.0; 3 * n_atoms];
        for &(i, j) in chunk {
            // evaluate pair, accumulate into local_e, local_f
        }
        (local_e, local_f)
    })
    .reduce(|| (0.0, vec![0.0; 3 * n_atoms]), |(e1, f1), (e2, f2)| {
        // sum energies and forces
    });
```

### 4. Avoid f64 <-> f32 Conversions

With `type F = f32` (default), avoid accidental promotion:
```rust
// BAD: 0.5 is f64, causes conversion
let half: F = 0.5;  // implicit f64->f32

// GOOD: explicit f32 literal
let half: F = 0.5 as F;
```

### 5. Minimize Branching in Kernels

```rust
// BAD: branch per pair
if dist < cutoff {
    energy += lj(dist);
}

// BETTER: filter before kernel, or use branchless mask
let mask = if dist < cutoff { 1.0 } else { 0.0 };
energy += mask * lj(dist);
```

### 6. CUDA Kernel Considerations

When writing CUDA kernels (`cuda` feature):
- Keep kernel launches coalesced (threads access consecutive memory)
- Minimize host<->device transfers
- Use shared memory for neighbor list tiles
- Prefer warp-level operations over atomic operations
- Match grid/block dimensions to problem size

## Benchmarking

```bash
# Run criterion benchmarks
cargo bench -p molrs-core
cargo bench -p molrs-md

# Run with specific benchmark
cargo bench -p molrs-core -- potential

# Generate flamegraph (requires cargo-flamegraph)
cargo flamegraph --bench potential -p molrs-core
```

### What to Benchmark

- Potential kernel eval time vs atom count (scaling test)
- Neighbor list build time vs atom count
- MD step time (all-inclusive)
- Packing objective+gradient evaluation time

### Performance Regression Rules

- New code must not regress existing benchmarks by > 5%
- New kernels must include criterion benchmark
- O(N^2) algorithms need justification (typically testing-only)

## Review Checklist

When reviewing performance-sensitive code:

- [ ] No allocation in inner loops
- [ ] Flat `&[F]` for coordinate access in kernels
- [ ] No unnecessary f64<->f32 conversions
- [ ] Rayon used where applicable (with `#[cfg(feature = "rayon")]`)
- [ ] No `BruteForce` neighbor list in production paths
- [ ] PairVisitor used for zero-allocation pair traversal
- [ ] SIMD-friendly loop structure (no branches)
- [ ] Benchmark included for new kernels
- [ ] No regression in existing benchmarks
- [ ] CUDA kernels use coalesced access patterns
