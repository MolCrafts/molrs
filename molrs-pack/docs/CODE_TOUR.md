# molrs-pack Code Tour

Navigation guide + extension tutorials for contributors. Companion: [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system-level design.

## 1. Where to start reading

Recommended order for a first read-through (≈ 2 hours):

1. **`lib.rs`** (73 lines) — see the public surface in one screen.
2. **`error.rs`** (41 lines) — error variants tell you what the crate cares about.
3. **`target.rs`** (241 lines) — `Target` is the user-facing data structure.
4. **`restraint.rs`** §1 only (trait def + 1 concrete `impl Restraint for InsideBoxRestraint`) — 100 lines. Don't read all 14 now.
5. **`handler.rs`** (378 lines) — observer pattern, easy to follow.
6. **`packer.rs::Molpack` struct + `Molpack::pack` (top 100 lines of fn)** — setup before the phase loop.
7. **`packer.rs::run_phase` and `run_iteration`** — the two levels of the phase loop.
8. **`objective.rs::compute_fg`** — the hot path that does all the work.
9. **`gencan/mod.rs::pgencan`** — the optimizer entry. Don't try to understand tn_ls/spg/cg on first pass; just trust they work.
10. **`region.rs`** — smaller than `restraint.rs` and self-contained. Good palate-cleanser before the deep dives.

**Skip on first pass**: `gencan/cg.rs`, `gencan/spg.rs`, `initial.rs`, `movebad.rs`. Come back when you need to touch them.

## 2. Trace of a `pack()` call

Walkthrough of what happens in sequence. Line numbers are approximate.

```
User: let result = Molpack::new().pack(&[target], 400, Some(42))?;

packer.rs:~203  Molpack::pack entry
   ├── empty / zero-atom / invalid-PBC checks         (early return on error)
   ├── BROADCAST global_restraints into each target   (Arc::clone per pair)
   ├── split into free / fixed targets
   │
   ├── PackContext::new(ntotat, ntotmol, ntype)        (allocate all buffers)
   ├── populate sys.restraints pool + CSR              (iratom_offsets/data)
   ├── place fixed atoms via eulerfixed + compcart
   ├── initial::initial(...)                           (random placement)
   ├── cell list build (latomfirst / latomnext)
   ├── handlers.on_start / on_initial
   │
   └── phase loop: for phase in 0..=ntype
          │
          packer.rs:~496  run_phase(phase, ...)
             ├── phase_info = PhaseInfo { phase, total_phases, mol_type }
             ├── handlers.on_phase_start(phase_info)
             ├── comptype reconfig (per-type vs all-types)
             ├── reset radscale → discale
             ├── swap.save_type if per-type phase
             ├── precision short-circuit (evaluate unscaled; return Converged if already under threshold)
             │
             └── inner loop: for loop_idx in 0..max_loops
                    │
                    packer.rs:~649  run_iteration(...)
                       ├── movebad if enabled and radscale==1 and stuck
                       ├── relaxer_runners loop:
                       │     for each (itype, runner):
                       │       saved = sys.coor[base..base+na].to_vec()
                       │       f_before = evaluate(FOnly)
                       │       runner.on_iter(saved, f_before, closure, rng)
                       │         — closure evaluates trial coords under full objective
                       │       accept/reject handled inside runner
                       ├── pgencan(x, &mut sys, params, precision, ws)
                       │     │
                       │     gencan/mod.rs::pgencan
                       │        ├── allocate bounds (obj.bounds(l, u))
                       │        ├── gencan(x, l, u, obj, ...)
                       │        │     │
                       │        │     tn_ls inner iterations
                       │        │       ├── obj.evaluate(x, FAndGradient, Some(g))
                       │        │       │     ↓
                       │        │       │  objective.rs:compute_fg
                       │        │       │     ├── expand_molecules (euler → xcart)
                       │        │       │     ├── accumulate_constraint_* per atom
                       │        │       │     ├── insert_atom_in_cell (build cell list)
                       │        │       │     ├── accumulate_pair_fg_parallel (rayon)
                       │        │       │     └── project_cartesian_gradient → g
                       │        │       ├── line search (alpha scaling)
                       │        │       └── cg/spg step
                       │        └── packmolprecision termination check
                       ├── evaluate_unscaled(sys, xwork)       (report fdist/frest under real radii)
                       ├── compute fimp = -100 * Δf/flast
                       ├── radscale decay (discale → 1.0)
                       ├── handlers.on_step(step_info, sys)
                       ├── converged check (frest < precision && fdist < precision)
                       ├── handlers.should_stop() → EarlyStop
                       └── IterOutcome::Continue | Converged | EarlyStop
                    │
             └── PhaseOutcome::Continue | Converged
          │
   ├── init_xcart_from_x() — final expansion
   ├── handlers.on_finish(sys)
   └── context_to_frame(sys) → PackResult { frame, converged, ... }
```

Useful tools for tracing live calls:

```bash
# Run the smallest example with progress logging
MOLRS_PACK_EXAMPLE_PROGRESS=1 cargo run -p molcrafts-molrs-pack --release --example pack_mixture

# Dump every handler step (requires attaching a custom Handler;
# see handler.rs::ProgressHandler for a template)
```

## 3. Tutorial: add a custom `Restraint`

Goal: add a restraint that pulls atoms toward a target plane with a quadratic attractive well (a "soft tether").

### Step 1: define the struct

In your own code (or in a branch of `molrs-pack/src/restraint.rs`):

```rust
use molrs::types::F;
use molrs_pack::Restraint;

#[derive(Debug, Clone, Copy)]
pub struct PlaneTether {
    pub normal: [F; 3],   // unit normal
    pub offset: F,         // n · x = offset defines the plane
    pub k: F,              // stiffness
}
```

Fields: pub, semantically named, no shared blob. Struct derives `Debug + Clone + Copy` — `Debug` is a trait supertrait bound so `Target`'s derived `Debug` keeps working. `Clone` is useful for `Molpack::add_restraint` broadcast semantics (not strictly required — `Arc` clones regardless).

### Step 2: implement `Restraint`

```rust
impl Restraint for PlaneTether {
    fn f(&self, pos: &[F; 3], _scale: F, _scale2: F) -> F {
        let d = self.normal[0] * pos[0]
              + self.normal[1] * pos[1]
              + self.normal[2] * pos[2]
              - self.offset;
        0.5 * self.k * d * d
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let d = self.normal[0] * pos[0]
              + self.normal[1] * pos[1]
              + self.normal[2] * pos[2]
              - self.offset;
        // gradient: k * d * n
        g[0] += self.k * d * self.normal[0];
        g[1] += self.k * d * self.normal[1];
        g[2] += self.k * d * self.normal[2];
        self.f(pos, scale, scale2)
    }
}
```

Three things worth noting:

1. **Gradient accumulates with `+=`.** Many restraints may touch the same atom; your `fg` must not overwrite.
2. **`fg` returns the value.** The hot path uses the returned value for the fdist/frest accumulation — don't return 0 just because the caller "might discard it".
3. **Scale/scale2 usage is your choice.** Packmol convention: linear-penalty restraints use `scale`; quadratic-penalty ones use `scale2`. Your tether is quadratic but uses its own `k` — you can ignore the scale knobs entirely.

### Step 3: write a gradient test

```rust
#[test]
fn plane_tether_gradient_matches_fd() {
    let r = PlaneTether { normal: [0.0, 0.0, 1.0], offset: 5.0, k: 2.0 };
    let x = [1.0, 2.0, 7.0];  // d = 2, violated
    let mut g = [0.0; 3];
    let _ = r.fg(&x, 1.0, 1.0, &mut g);

    let h: F = 1e-5;
    for k in 0..3 {
        let mut xp = x; xp[k] += h;
        let mut xm = x; xm[k] -= h;
        let fd = (r.f(&xp, 1.0, 1.0) - r.f(&xm, 1.0, 1.0)) / (2.0 * h);
        assert!((g[k] - fd).abs() < 1e-4, "axis {k}: analytic={}, fd={}", g[k], fd);
    }
}
```

Convention: ε=1e-5, tolerance=1e-3 (tighter if your restraint is smoother, looser if it has kinks).

### Step 4: use it

```rust
let target = Target::from_coords(&pos, &radii, 100)
    .with_restraint(PlaneTether { normal: [0.0, 0.0, 1.0], offset: 5.0, k: 2.0 });
```

Your type composes with built-ins identically:

```rust
let target = Target::from_coords(&pos, &radii, 100)
    .with_restraint(InsideBoxRestraint::new([0.0; 3], [40.0; 3]))
    .with_restraint(PlaneTether { normal: [0.0, 0.0, 1.0], offset: 20.0, k: 1.0 });
```

This is direction 3 in action — built-in `InsideBoxRestraint` and user `PlaneTether` go through the same code path.

## 4. Tutorial: add a composite `Region`

Goal: a conical region (inside a cone with apex at origin, axis along +z, half-angle 30°).

### Step 1: define the struct + impl Region

```rust
use molrs::types::F;
use molrs_pack::Region;

#[derive(Debug, Clone, Copy)]
pub struct ConeRegion {
    pub apex: [F; 3],
    pub axis: [F; 3],        // unit vector
    pub half_angle_cos: F,   // cos(half-angle), precomputed
}

impl Region for ConeRegion {
    fn contains(&self, x: &[F; 3]) -> bool {
        self.signed_distance(x) <= 0.0
    }

    fn signed_distance(&self, x: &[F; 3]) -> F {
        let dx = x[0] - self.apex[0];
        let dy = x[1] - self.apex[1];
        let dz = x[2] - self.apex[2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        if r < 1e-12 { return 0.0; }
        let axis_dot = (dx*self.axis[0] + dy*self.axis[1] + dz*self.axis[2]) / r;
        // signed distance on the cone surface: angular, not metric.
        // negative when axis_dot >= cos(half_angle), i.e. inside cone.
        self.half_angle_cos - axis_dot
    }

    // Use default FD gradient for now — override for hot-path use.
}
```

### Step 2: compose with built-ins

```rust
use molrs_pack::{FromRegion, InsideSphereRegion, RegionExt};

// Cone ∩ sphere shell — a "light cone" inside a sphere
let cone = ConeRegion {
    apex: [0.0; 3],
    axis: [0.0, 0.0, 1.0],
    half_angle_cos: (std::f64::consts::PI / 6.0).cos(),
};
let sphere = InsideSphereRegion::new([0.0; 3], 10.0);

let region = cone.and(sphere);
let target = Target::from_coords(&pos, &radii, 100)
    .with_restraint(FromRegion(region));
```

`RegionExt::and/or/not` come from a blanket impl on every `Region`. The resulting type is `And<ConeRegion, InsideSphereRegion>` — static dispatch, no heap.

### Step 3: analytic gradient override (for hot-path correctness)

If you deploy a composite region for real packing (not just prototyping), override the default FD `signed_distance_grad`. The cone above:

```rust
fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
    let dx = x[0] - self.apex[0];
    let dy = x[1] - self.apex[1];
    let dz = x[2] - self.apex[2];
    let r2 = dx*dx + dy*dy + dz*dz;
    let r = r2.sqrt();
    if r < 1e-12 { return [0.0; 3]; }
    // d(axis_dot)/dx = (axis_k * r - (dx · axis) * dx_k / r) / r
    //                = axis_k / r - axis_dot * dx_k / r
    let axis_dot = (dx*self.axis[0] + dy*self.axis[1] + dz*self.axis[2]) / r;
    // signed_distance = cos(α) - axis_dot  ⇒  grad = -∂axis_dot/∂x
    let inv_r = 1.0 / r;
    [
        -(self.axis[0] * inv_r - axis_dot * dx * inv_r * inv_r),
        -(self.axis[1] * inv_r - axis_dot * dy * inv_r * inv_r),
        -(self.axis[2] * inv_r - axis_dot * dz * inv_r * inv_r),
    ]
}
```

Then FD-check it as in §3 step 3.

## 5. Tutorial: add a custom `Handler`

Goal: a handler that writes a CSV row per step so you can plot objective evolution.

```rust
use std::fs::File;
use std::io::{BufWriter, Write};
use molrs_pack::{F, Handler, PackContext, StepInfo};

pub struct CsvHandler {
    writer: BufWriter<File>,
}

impl CsvHandler {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        writeln!(w, "phase,loop_idx,fdist,frest,improvement_pct")?;
        Ok(Self { writer: w })
    }
}

impl Handler for CsvHandler {
    fn on_step(&mut self, info: &StepInfo, _sys: &PackContext) {
        let _ = writeln!(
            self.writer,
            "{},{},{},{},{}",
            info.phase.phase, info.loop_idx, info.fdist, info.frest, info.improvement_pct
        );
    }
}

// Attach:
let packer = Molpack::new().add_handler(CsvHandler::new("trace.csv")?);
```

Handler notes:

- **`on_step` is the only required method.** Everything else has a default no-op.
- **Observers are `&mut self` but `sys` is `&PackContext`**: you cannot mutate the packer's state from a handler. Use a `Relaxer` if you need to modify atom positions.
- **Multiple handlers run in registration order.** `add_handler` chains — register your CSV handler before `ProgressHandler` to get rows even on the last step, vice-versa otherwise.
- **`should_stop` is polled every iteration.** Return `true` to break the outer loop early. Useful for time budgets or convergence criteria other than `precision`.

## 6. Tutorial: add a custom `Relaxer`

Goal: a relaxer that tries a random rigid-body translation of the whole molecule and accepts if the objective decreases.

```rust
use molrs::types::F;
use molrs_pack::{Relaxer, RelaxerRunner};
use rand::rngs::SmallRng;
use rand::{RngCore, Rng};

#[derive(Debug, Clone)]
pub struct JiggleRelaxer {
    pub steps: usize,
    pub max_delta: F,
}

impl Relaxer for JiggleRelaxer {
    fn build(&self, _ref_coords: &[[F; 3]]) -> Box<dyn RelaxerRunner> {
        Box::new(JiggleRunner {
            steps: self.steps,
            max_delta: self.max_delta,
            accepted: 0,
            total: 0,
        })
    }
}

pub struct JiggleRunner {
    steps: usize,
    max_delta: F,
    accepted: usize,
    total: usize,
}

impl RelaxerRunner for JiggleRunner {
    fn on_iter(
        &mut self,
        coords: &[[F; 3]],
        f_current: F,
        evaluate: &mut dyn FnMut(&[[F; 3]]) -> F,
        rng: &mut dyn RngCore,
    ) -> Option<Vec<[F; 3]>> {
        let mut best = coords.to_vec();
        let mut best_f = f_current;
        let mut accepted_any = false;

        for _ in 0..self.steps {
            // Random translation via rand::distributions trick
            let jx = (rng.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) * self.max_delta;
            let jy = (rng.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) * self.max_delta;
            let jz = (rng.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) * self.max_delta;
            let trial: Vec<[F; 3]> = best.iter().map(|p| [p[0]+jx, p[1]+jy, p[2]+jz]).collect();
            let f_trial = evaluate(&trial);
            self.total += 1;
            if f_trial < best_f {
                self.accepted += 1;
                best_f = f_trial;
                best = trial;
                accepted_any = true;
            }
        }

        if accepted_any { Some(best) } else { None }
    }

    fn acceptance_rate(&self) -> F {
        if self.total == 0 { 0.0 } else { self.accepted as F / self.total as F }
    }
}

// Attach (note: count must be 1 because relaxer modifies ref coords shared
// across copies — this is the Packmol-inherited constraint):
let target = Target::from_coords(&polymer_coords, &radii, 1)
    .with_relaxer(JiggleRelaxer { steps: 10, max_delta: 0.5 });
```

Relaxer notes:

- **`Relaxer` is the immutable builder; `RelaxerRunner` holds per-pack state** (acceptance counters, temperature, etc.). `build()` is called once per target type at `pack()` entry.
- **`evaluate` closure tests trial coords against the full objective** without mutating the reference — use it as often as you like.
- **Return `Some(new_coords)` only if you actually changed something.** The packer skips unnecessary cache invalidation when you return `None`.
- **`count == 1` constraint**: all copies of a molecule type share the same `ref_coords`. If you mutate them, all copies change. Multi-copy targets must not have relaxers.

## 7. Testing discipline

| Kind | Where | Example |
|---|---|---|
| Unit test for a function | Inside `#[cfg(test)] mod tests` in the same file | `region.rs` has its own test module |
| Integration test for a public API | `tests/<name>.rs` | `tests/packer.rs` for `Molpack::pack` |
| Gradient finite-difference | Inside or alongside the unit test | `tests/gradient.rs` |
| Regression against Packmol | `tests/examples_batch.rs` `#[ignore]` | Run with `--ignored` in release |
| Criterion bench | `benches/<name>.rs` | `benches/pack_end_to_end.rs` |

Rules:

- **Every new `Restraint` gets an FD gradient test.** ε=1e-5, tol=1e-3.
- **Every new `Region` gets boolean-algebra tests on its combinators and a signed-distance sign test.** Look at `region.rs::tests` for the pattern.
- **Every hot-path change gets a microbench in `benches/`.** Follow the `fn` + `caller` two-bench pattern in the existing bench files.
- **Don't write tests against the old Constraint names.** They do not exist.

Run all tests:

```bash
cargo test -p molcrafts-molrs-pack --all-features
cargo test -p molcrafts-molrs-pack --release --test examples_batch -- --ignored
```

## 8. Benchmarking discipline

End-to-end alarm (catastrophic-regression gate):

```bash
cargo bench -p molcrafts-molrs-pack --bench pack_end_to_end -- mixture
# Workloads: mixture (~2s), bilayer (~10s), interface (~5s), solvprotein (~30s),
# spherical (~45 min with 18k molecules).
```

Microbenches (fast, run after any hot-path edit):

```bash
cargo bench -p molcrafts-molrs-pack --bench evaluate_unscaled
cargo bench -p molcrafts-molrs-pack --bench run_iteration
cargo bench -p molcrafts-molrs-pack --bench run_phase
cargo bench -p molcrafts-molrs-pack --bench objective_dispatch
```

Gates (from `.claude/specs/molrs-pack-plugin-arch.md` §10):

| Scope | Hard gate | Soft gate |
|---|---|---|
| Per-function microbench | ≤ +1% | ≤ 0% |
| Caller microbench (includes indirection) | ≤ +2% | ≤ +1% |
| `pack_end_to_end` (phase-end alarm) | ≤ +10% | ≤ +5% |

If you cross the soft gate, attach a flamegraph + one-paragraph root-cause note to the PR. If you cross the hard gate, don't merge — tighten the change or roll back.

**Extract-bench pattern** (for hot-path refactors only):

1. Add a `#[cfg(bench)] #[inline(never)] fn F_sentinel(...)` holding the pre-extraction body.
2. Add microbench of the extracted fn + caller.
3. Make the extraction in the same commit.
4. Delete the sentinel at the END of the next refactor cycle (not sooner).

Phase A used this for `evaluate_unscaled` / `run_iteration` / `run_phase`; the sentinels were deleted at end-of-phase-B.

## 9. Common pitfalls

- **Gradient sign direction.** Every `Restraint` accumulates `∂penalty/∂x`. The optimizer MINIMIZES the objective and negates for descent. If your molecules fly out of the region, your gradient has the wrong sign — the penalty should point toward the violation boundary.
- **Rotation convention.** Single-atom unit tests pass with both LEFT and RIGHT Euler multiplication; multi-atom tests don't. Always test your Euler changes against a molecule with ≥ 2 atoms.
- **`Cell<f64>` is not `Sync`.** If you need interior mutability in a `Send + Sync` context, use `AtomicU64` + `f64::to_bits`/`f64::from_bits`.
- **`Target::with_restraint_for_atoms` uses 1-based indexing.** Packmol convention. `&[1, 2]` selects the first two atoms. Internally converted to 0-based.
- **`count = 1` required for relaxers.** Multi-copy targets share reference coords; a relaxer mutating them would change all copies silently.
- **`radscale` changes per phase.** Don't hard-code atomic radii — always go through `sys.radius[i]`. `evaluate_unscaled` temporarily swaps `radius` with `radius_ini` to report pre-scale numbers to the user.
- **PBC boxes must be valid.** `Molpack::pbc(min, max)` with any zero-length axis returns `PackError::InvalidPBCBox`. Use `Molpack::pbc_box(lengths)` for axis-aligned origin-at-zero.

## 10. When you're stuck

- Check `.claude/specs/molrs-pack-plugin-arch.md` first. The spec is the source of truth for all current design decisions.
- Check `docs/packmol_alignment.md` for the Fortran ↔ Rust kind-number mapping.
- For physics questions (two-scale contract, rotation LEFT/RIGHT, gradient sign), look at comments referencing `comprest.f90` / `gwalls.f90` / `computef.f90` — those point at the exact Fortran lines being ported.
- For "why is this slow?" questions, start with `objective.rs` — that's the hot path. `cargo flamegraph` + `pack_end_to_end/mixture` gives a solid first-pass profile.

## 11. Contributing flow

1. Open the spec (`molrs-pack-plugin-arch.md`) and find the checklist entry closest to your change. If there isn't one, propose a spec edit first.
2. Write a failing test (unit or integration).
3. Implement until the test passes.
4. Run the full gate: `cargo test --all-features && cargo clippy -- -D warnings && cargo fmt --check`.
5. If hot-path: run the relevant microbench and `pack_end_to_end/mixture` before and after. Attach numbers to the PR.
6. Update the spec's Commit log section.

See `.claude/skills/molrs-impl/SKILL.md` for the `/molrs-impl` automation that chains these steps.
