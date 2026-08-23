# Performance Standards

Project standard for molrs performance. Applied by the `mol:optimizer` agent
and `/mol:review`.

## Hot Path Hierarchy (most → least critical)

1. **Pair potential evaluation** — O(N·k) with neighbor lists; called every MD step.
2. **Neighbor list build/update** — O(N) with `LinkCell`; rebuilt periodically.
3. **Force accumulation** — sum across all potential terms.
4. **GENCAN inner loop** — objective + gradient evaluation (now in the
   standalone `MolCrafts/molpack` repo; kept here for historical context).

Known hot-path files: `molrs/src/ff/potential/**`, `molrs/src/core/neighbors/**`
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

`LinkCell` (default) is O(N) build + O(N·k) traversal. Rules:

- Cell size ≥ cutoff so only 27 neighboring cells are scanned.
- Use `PairVisitor` callback for zero-allocation traversal.
- Apply Verlet skin distance — do **not** rebuild every step.
- `BruteForce` is O(N²) — testing only, never production.
- Rayon-parallel build is feature-gated.

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

## Benchmarking

Two bench targets exist, both criterion (`harness = false`):
`core_benchmarks` (45 benchmarks) and `compute_benchmarks` (53, gated on the
`compute` feature). There is **no** `potential` target — the commands below are
the ones that actually run, each verified 2026-08-23.

```bash
# See what exists before filtering — ids are criterion group paths
cargo bench -p molcrafts-molrs --bench core_benchmarks -- --list
cargo bench -p molcrafts-molrs --bench compute_benchmarks --features compute -- --list

# Whole target (this is the canonical one, same as CLAUDE.md and bench.yml)
cargo bench -p molcrafts-molrs --bench core_benchmarks

# Filter by group id. Real ids include frame/get, graph/find_rings,
# neighbors/build, core/simbox/shortest_vector, ndarray_vs_vec/matmul.
cargo bench -p molcrafts-molrs --bench core_benchmarks -- 'neighbors/build'

# Profile one target (needs `cargo install flamegraph`)
cargo flamegraph --bench core_benchmarks -p molcrafts-molrs

# Local comparisons only — never for numbers you intend to record, since
# target-cpu=native is not what CI or a release build uses
RUSTFLAGS="-C target-cpu=native" cargo bench -p molcrafts-molrs --bench core_benchmarks
```

What to benchmark — a wish list, **not** an inventory. Only the second line is
covered today (`neighbors/*` in `core_benchmarks`); the other two have no
benchmark in either target, so do not go looking for one:

- Potential kernel eval vs atom count (scaling) — *not covered*
- Neighbor list build vs atom count — covered
- Full MD step (all-inclusive) — *not covered*
- GENCAN objective + gradient eval (molpack repo)

## Benchmarking during refactors

When extracting pure functions from a monolith (common in `molrs/src/ff/`
kernel extractions and `molrs/src/core/` neighbor rewrites — modules, not
crates, since the single-crate merge; historically also the objective/packer
splits now living in molpack), **do not rely on end-to-end
benchmarks to guard the extraction**. Macro-level noise (allocator, cache, OS
scheduler) routinely produces ±3% drift independent of the change, which masks
real per-function regressions and makes tight gates statistically meaningless.

**Discipline per extraction** (every extracted function is its own atomic commit):

1. Before moving the function body, add `#[cfg(bench)] #[inline(never)] fn
   F_sentinel(...)` holding the pre-extraction behavior in the same module. The
   sentinel compiles only under the `bench` cfg and does not ship in release
   binaries.
2. In the same commit as the extraction, land:
   - A unit test pinning the function's observable behavior against the
     sentinel (or against a known-good reference).
   - A criterion microbench of the extracted function.
   - A criterion microbench of the *caller* (catches indirection / vtable /
     inlining-boundary costs that the function microbench cannot see).
3. Gates:
   - Function microbench: extracted ≤ +1% vs. `F_sentinel`, or improvement.
   - Caller microbench: ≤ +2% vs. pre-extraction snapshot.
4. Delete `F_sentinel` once the extraction has been on main for one follow-up
   refactor cycle. The microbench itself stays permanently as a
   future-regression guard.

**End-to-end benches** are kept only as a **catastrophic-regression alarm**
(> 10% blocks the merge). They are not the primary gate for per-extraction
changes. A single representative end-to-end bench per crate is usually enough
for this role — do not build matrix baselines (`{1k,10k,100k} × {1,4,16}`) as
per-PR gates; the signal-to-noise ratio does not justify the runtime cost.

Rationale: a 17 ms monolithic bench cannot localize which of 12 extracted
functions regressed 2%. Per-function microbenches can. This matters whenever
the refactor goal states "zero performance loss".

Do **not** approve an extraction on the basis of an end-to-end bench alone —
its noise floor (±3%) is the same magnitude as the gate and carries no
per-function signal.

## Performance Budget

- Newly-extracted pure functions MUST include a criterion microbench in the
  same commit as the extraction. New kernels MUST include a criterion
  microbench.
- Per-extraction microbench gate: extracted ≤ +1% vs. `#[inline(never)]`
  sentinel; caller ≤ +2%.
- End-to-end bench gate: ≤ +10% (catastrophic alarm only); sustained
  incremental drift above +5% requires a profiler report and HPC review.
- O(N²) algorithms require justification (testing-only).

## Compliance Checklist

- [ ] No allocation in inner loops
- [ ] Flat `&[F]` for kernel coordinate access
- [ ] No spurious `f32 ↔ f64` conversions
- [ ] Rayon used where applicable, gated on `#[cfg(feature = "rayon")]`
- [ ] No `BruteForce` in production paths
- [ ] `PairVisitor` used for pair traversal
- [ ] SIMD-friendly loop structure (no branches)
- [ ] Benchmark included for new kernels
- [ ] No regression in existing benchmarks
- [ ] For extractions from hot-path monoliths: `#[cfg(bench)]
      #[inline(never)]` sentinel + function microbench + caller microbench all
      landed in the same commit as the extraction
- [ ] Per-extraction microbench gate ≤ +1% vs. sentinel; caller gate ≤ +2%
- [ ] End-to-end catastrophic alarm bench present where monolithic paths still
      exist (≤ +10% gate)
