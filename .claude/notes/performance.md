# Performance Standards

Project standard for molrs performance. Applied by the `mol:optimizer` agent
and `/mol:review`.

## Hot Path Hierarchy (most → least critical)

1. **Pair potential evaluation** — O(N·k) with neighbor lists; called every MD step.
2. **Neighbor list build/update** — O(N) with `LinkCell`; rebuilt periodically.
3. **Force accumulation** — sum across all potential terms.
4. **GENCAN inner loop** — objective + gradient evaluation (now in the
   standalone `MolCrafts/molpack` repo; kept here for historical context).

Known hot-path files: `molrs/src/ff/potential/**`, `molrs/src/core/spatial/neighbors/**`
(plus `objective.rs` / `packer.rs` in the molpack repo).

## Memory Layout

Prefer Structure-of-Arrays (SoA) over Array-of-Structures (AoS):

```rust
// GOOD (SoA) — cache-friendly per-component access
let x: Array1<F> = ...;
let y: Array1<F> = ...;
let z: Array1<F> = ...;

// ACCEPTABLE — row-major Array2
let coords: Array2<F> = Array2::zeros((n_atoms, 3));

// BAD — pointer chasing
let atoms: Vec<Atom> = ...;
```

The Zarr trajectory format and `Block` are SoA by design.

### Flat coordinate vectors for kernels

Potential kernels use flat `&[F]` (3N elements): `[x0,y0,z0, x1,y1,z1, ...]`.
Enables contiguous access and auto-vectorization.

## Neighbor Lists

Three backends behind the `NeighborList` engine, picked by constructor:

| Backend | Build | Use when |
|---|---|---|
| `LinkCell` (`NeighborList::new`) | O(N) | **Default.** Roughly uniform density and one dominant cutoff. |
| `Aabb` (`NeighborList::aabb`) | O(N log N) | Non-uniform density, or particle sizes spread wide enough that one cutoff makes cells coarse. |
| `BruteForce` (`NeighborList::brute_force`) | O(N²) | Test oracle only, never production. |

Rules:

- `LinkCell` cell size ≥ cutoff so only 27 neighboring cells are scanned. A
  tilted cell must size from `SimBox::nearest_plane_distance`, never
  `lengths()` — `|a_k|` over-estimates the width and pairs go missing.
- Use `PairVisitor` callback for zero-allocation traversal.
- Apply Verlet skin distance — do **not** rebuild every step.
- Rayon-parallel build is feature-gated.

**Owed measurement.** The `LinkCell`-vs-`Aabb` crossover above is the
freud-derived rationale, not a molrs measurement — treat the "use when" column
as a hypothesis and keep `LinkCell` as the default.

## Optimization Rules

### 1. No allocation in inner loops

Reuse buffers via `&mut` parameter or stored field. `Vec::with_capacity` when
size is known.

### 2. SIMD-friendly patterns

```rust
// GOOD — vectorizable
for i in 0..n {
    forces[3*i]   += scale * dx;
    forces[3*i+1] += scale * dy;
    forces[3*i+2] += scale * dz;
}

// BAD — branch in hot loop
for i in 0..n {
    if atoms[i].is_active { ... }
}
```

### 3. Rayon parallelism

Pattern: parallel reduce with thread-local accumulator.

```rust
use rayon::prelude::*;

let (energy, forces) = pairs.par_chunks(chunk_size)
    .map(|chunk| {
        let mut local_e = 0.0_f64;
        let mut local_f = vec![0.0_f64; 3 * n_atoms];
        for &(i, j) in chunk { /* ... */ }
        (local_e, local_f)
    })
    .reduce(
        || (0.0_f64, vec![0.0_f64; 3 * n_atoms]),
        |(e1, mut f1), (e2, f2)| {
            for k in 0..f1.len() { f1[k] += f2[k]; }
            (e1 + e2, f1)
        });
```

### 4. Float literals

`F = f64` always. Use plain `0.5`, `2.0` literals (already `f64`). Avoid
`as F` casts and `f32 ↔ f64` conversions.

### 5. Branchless when cheap

```rust
// BAD — branch per pair
if dist < cutoff { energy += lj(dist); }

// BETTER — pre-filter or branchless mask
let mask = (dist < cutoff) as u32 as F;
energy += mask * lj(dist);
```

## Measurement

There are no benchmark targets in this repo; the benchmark and regression
systems are being redesigned. Until they land, a performance change is
verified for **correctness only** (the unit suite) and its speed claim is
recorded as owed, not asserted.

## Owed (2026-09-20)

Hot-path findings recorded during the 0.14 cleanup and not yet acted on:

- `core/spatial/neighbors/mod.rs` — `Backend::visit_pairs(&mut dyn PairVisitor)`
  makes `for_each_pair` pay a virtual call per pair; add a monomorphised
  `visit_pairs_with<V: PairVisitor>` sibling.
- `core/spatial/neighbors/verlet_skin.rs` — the pass-two table fill is serial
  per step for edge order; the fixed-chunk `fold_chunks` pattern from
  `ff/potential/pair/mod.rs` would keep the order and parallelise it.
- `ff/potential/kspace/pme.rs` — `Mutex<PmeScratch>` serialises every
  evaluation; use `&mut self` or thread-local scratch. PME and every bonded
  kernel are sequential while offline `compute/` has 60 rayon sites.
- `ff/potential/pair/mod.rs` — `fold_chunks` allocates one full-width buffer
  per chunk; `map_init` (as in `compute/voronoi/radical.rs`) keeps the
  semantics without the allocations.
- `core/spatial/neighbors/{mod,aabb,filter}.rs` — no `with_capacity` on
  materialisation, a `HashMap` per k-NN query, one `Vec` per query point in
  the filter; CSR as in `compute/cluster/mod.rs`.
- `compute/distribution/observable.rs`, `compute/voronoi/radical.rs`,
  `compute/diffraction/diffraction_pattern.rs`, `compute/msd/mod.rs` — per-frame
  column copies, a `Vec` per spatial bin, an `FftPlanner` per call.
- `md/forces.rs` — `ForceOutput` owns its forces, so `MicPairs` clones its
  accumulator every step.
- `md/` (5 k lines) has no rayon at all.
- Three minimum-image implementations (`verlet_skin.rs`, `linkcell.rs`,
  `pme.rs`); unify on `SimBox::mic()`.
- O(N²) loops without a stated justification:
  `compute/environment/angular_separation.rs`, `compute/voronoi/radical.rs`,
  `compute/diffraction/direct.rs`.

## Compliance Checklist

- [ ] No allocation in inner loops
- [ ] Flat `&[F]` for kernel coordinate access
- [ ] No spurious `f32 ↔ f64` conversions
- [ ] Rayon used where applicable, gated on `#[cfg(feature = "rayon")]`
- [ ] No `BruteForce` in production paths
- [ ] `PairVisitor` used for pair traversal
- [ ] SIMD-friendly loop structure (no branches)
- [ ] Kernels fold into the caller's buffer (`Potential::accumulate`), not a
      fresh `Vec` per call
