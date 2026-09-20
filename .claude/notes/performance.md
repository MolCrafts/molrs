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

Hot-path findings from the 0.14 cleanup, with what was done about each.

**Fixed**

- `md/forces.rs` — `ForceProvider::compute_into(pos, shifts, &mut ForceOutput)`:
  `MicPairs` swaps its accumulator with the caller's array instead of cloning
  it; `GhostPairs` copies the owned prefix into the caller's array (its
  accumulator also covers the copies); the integrators lend `state.forces`
  and take it back, so a step allocates nothing. `compute()` stays as the
  allocating convenience.
- `ff/potential/kspace/pme.rs` — the exclusion-correction minimum image goes
  through `SimBox`'s `Mic` (orthogonal fast path, triclinic general form);
  PME no longer carries its own. `Mic` and `SimBox::shortest_vector_impl`
  are one kernel, so there is one minimum-image implementation left.
- `core/spatial/neighbors/filter.rs` — pairs are grouped by query point with
  a counting sort (CSR), no `HashMap<u32, Vec>` per call.
- `core/spatial/neighbors/aabb.rs` — `query_knn` collects `(point, d²)` hits
  into one Vec and reduces by sort + dedup; no `HashMap` per query, and equal
  distances tie-break on the point index (deterministic where the map was not).
- `compute/voronoi/radical.rs` — the cell grid is CSR (`cell_start` /
  `members`): one allocation instead of one Vec per bin, same iteration order.
- `compute/environment/angular_separation.rs`, `compute/diffraction/direct.rs`
  — the quadratic loops state their justification (the observable *is* the
  full matrix; the direct estimator is defined as the sum).

**Decided, not changed**

- `Backend::visit_pairs(&mut dyn PairVisitor)` — the backend sits behind
  `Box<dyn Backend>`, so a monomorphised visitor needs an enum-dispatched
  backend. The indirect call is ~1 ns against a ~10 ns MIC-plus-distance body
  and only the streaming path (`for_each_pair`) pays it; materialisation is
  unaffected. Not worth an enum until a measurement says so.
- `verlet_skin.rs` pass two — serial by contract: rows come out in edge order.
  A chunked parallel fill needs a `Neighbors` API that writes rows at fixed
  offsets; the pass is one branch per edge and pass one (the geometry) is
  already parallel.
- `pme.rs` `Mutex<PmeScratch>` — `Potential` evaluates through `&self`; the
  lock is uncontended (one evaluation at a time) and costs one atomic per
  call. Thread-local scratch buys nothing until PME evaluations run
  concurrently.
- `pair/mod.rs` `fold_chunks` — the per-chunk buffers are the point: partial
  sums merge in chunk order so the answer does not depend on the thread
  count. `map_init` would let a worker fold several chunks into one buffer in
  scheduling order and break that.
- `compute/distribution/observable.rs`, `compute/msd/mod.rs` — the copies are
  one-time (index build, reference frame), not per frame;
  `compute/diffraction/diffraction_pattern.rs` — a `FftPlanner` per call is
  microseconds against an n×n FFT.
- `core/spatial/neighbors/mod.rs` `Neighbors::empty` — there is no size hint
  before the search; a guessed capacity is a guess.

**Open (needs its own spec)**

- `md/` has no rayon: bonded kernels and PME are sequential. Parallelising
  the MD force path is a design task (per-thread accumulators, deterministic
  reduction), not a local fix.

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
