# Extending the Crate

Tutorials for writing your own `Restraint` / `Region` / `Handler` /
`Relaxer` types. Every extension trait in this crate follows the same
shape (direction-3 rule — see [`concepts`](crate::concepts)):

> `pub trait X` + N concrete `pub struct` types implementing it.
> User types `impl X` identically. No built-in/plugin type-level
> distinction.

## Table of contents

1. [Custom `Restraint`](#custom-restraint)
2. [Custom `Region`](#custom-region)
3. [Custom `Handler`](#custom-handler)
4. [Custom `Relaxer`](#custom-relaxer)
5. [Testing discipline](#testing-discipline)
6. [Benchmarking discipline](#benchmarking-discipline)
7. [Common pitfalls](#common-pitfalls)
8. [Contributing flow](#contributing-flow)

## Custom `Restraint`

Goal: pull atoms toward a target plane with a quadratic attractive
well.

### Step 1 — define the struct

```rust
use molrs::types::F;
# use molrs_pack::Restraint;

#[derive(Debug, Clone, Copy)]
pub struct PlaneTether {
    pub normal: [F; 3],
    pub offset: F,
    pub k: F,
}
# impl Restraint for PlaneTether {
#     fn f(&self, _x: &[F; 3], _s: F, _s2: F) -> F { 0.0 }
#     fn fg(&self, _x: &[F; 3], _s: F, _s2: F, _g: &mut [F; 3]) -> F { 0.0 }
# }
```

- `pub` fields — users construct with `PlaneTether { normal, offset,
  k }`, no builder.
- `Debug` required because [`Restraint`](crate::Restraint) has a
  `Debug` supertrait bound (so `Target`'s derived `Debug` keeps
  working).

### Step 2 — implement `Restraint`

```rust
# use molrs::types::F;
# use molrs_pack::Restraint;
# #[derive(Debug)]
# pub struct PlaneTether { pub normal: [F; 3], pub offset: F, pub k: F }
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
        g[0] += self.k * d * self.normal[0];
        g[1] += self.k * d * self.normal[1];
        g[2] += self.k * d * self.normal[2];
        self.f(pos, scale, scale2)
    }
}
```

Three contracts that all restraints must obey:

1. **Gradient accumulates with `+=`.** Multiple restraints may touch
   the same atom.
2. **`fg` returns the value.** The hot path uses the returned value
   for the `fdist`/`frest` accumulation — don't return `0.0` just
   because the caller might discard it.
3. **Scale/scale2 usage is your choice.** Linear-penalty restraints
   typically use `scale`; quadratic-penalty ones use `scale2`. Your
   tether uses its own `k` — ignore both knobs if you prefer.

### Step 3 — write a gradient test

```rust,no_run
# use molrs::types::F;
# use molrs_pack::Restraint;
# #[derive(Debug)] pub struct PlaneTether { pub normal: [F; 3], pub offset: F, pub k: F }
# impl Restraint for PlaneTether {
#     fn f(&self, _x: &[F; 3], _s: F, _s2: F) -> F { 0.0 }
#     fn fg(&self, _x: &[F; 3], _s: F, _s2: F, _g: &mut [F; 3]) -> F { 0.0 }
# }
#[test]
fn plane_tether_gradient_matches_fd() {
    let r = PlaneTether { normal: [0.0, 0.0, 1.0], offset: 5.0, k: 2.0 };
    let x = [1.0, 2.0, 7.0];
    let mut g = [0.0; 3];
    let _ = r.fg(&x, 1.0, 1.0, &mut g);

    let h: F = 1e-5;
    for k in 0..3 {
        let mut xp = x; xp[k] += h;
        let mut xm = x; xm[k] -= h;
        let fd = (r.f(&xp, 1.0, 1.0) - r.f(&xm, 1.0, 1.0)) / (2.0 * h);
        assert!(
            (g[k] - fd).abs() < 1e-4,
            "axis {k}: analytic={}, fd={}", g[k], fd,
        );
    }
}
```

Convention: `ε = 1e-5`, tolerance `1e-3` (looser if the restraint has
kinks).

### Step 4 — use it

```rust,no_run
# use molrs::types::F;
# use molrs_pack::Restraint;
# #[derive(Debug, Clone, Copy)] pub struct PlaneTether { pub normal: [F; 3], pub offset: F, pub k: F }
# impl Restraint for PlaneTether {
#     fn f(&self, _x: &[F; 3], _s: F, _s2: F) -> F { 0.0 }
#     fn fg(&self, _x: &[F; 3], _s: F, _s2: F, _g: &mut [F; 3]) -> F { 0.0 }
# }
use molrs_pack::{InsideBoxRestraint, Target};
# let (pos, rad) = (&[[0.0; 3]][..], &[1.0][..]);

let target = Target::from_coords(pos, rad, 100)
    .with_restraint(InsideBoxRestraint::new([0.0; 3], [40.0; 3]))
    .with_restraint(PlaneTether { normal: [0.0, 0.0, 1.0], offset: 20.0, k: 1.0 });
```

Built-in `InsideBoxRestraint` and user `PlaneTether` take the same
code path — direction-3 in action.

## Custom `Region`

Goal: a conical region with apex at origin, axis along +z,
half-angle 30°.

```rust
use molrs::types::F;
# use molrs_pack::Region;

#[derive(Debug, Clone, Copy)]
pub struct ConeRegion {
    pub apex: [F; 3],
    pub axis: [F; 3],
    pub half_angle_cos: F,
}

impl Region for ConeRegion {
    fn contains(&self, x: &[F; 3]) -> bool {
        self.signed_distance(x) <= 0.0
    }

    fn signed_distance(&self, x: &[F; 3]) -> F {
        let dx = x[0] - self.apex[0];
        let dy = x[1] - self.apex[1];
        let dz = x[2] - self.apex[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 1e-12 { return 0.0; }
        let axis_dot =
            (dx * self.axis[0] + dy * self.axis[1] + dz * self.axis[2]) / r;
        self.half_angle_cos - axis_dot
    }
    // Default FD gradient is OK for prototypes. Override analytically
    // for hot-path use — see below.
}
```

### Compose with built-ins

```rust,no_run
# use molrs::types::F;
# use molrs_pack::Region;
# #[derive(Debug, Clone, Copy)]
# pub struct ConeRegion { pub apex: [F; 3], pub axis: [F; 3], pub half_angle_cos: F }
# impl Region for ConeRegion {
#     fn contains(&self, _x: &[F; 3]) -> bool { true }
#     fn signed_distance(&self, _x: &[F; 3]) -> F { 0.0 }
# }
use molrs_pack::{FromRegion, InsideSphereRegion, RegionExt, Target};
# let (pos, rad) = (&[[0.0; 3]][..], &[1.0][..]);

let cone = ConeRegion {
    apex: [0.0; 3],
    axis: [0.0, 0.0, 1.0],
    half_angle_cos: (std::f64::consts::PI / 6.0).cos(),
};
let sphere = InsideSphereRegion::new([0.0; 3], 10.0);
let region = cone.and(sphere);

let target = Target::from_coords(pos, rad, 100)
    .with_restraint(FromRegion(region));
```

[`RegionExt::and`](crate::RegionExt::and) / `or` / `not` come from a
blanket impl on every `Region`. The resulting type
`And<ConeRegion, InsideSphereRegion>` is static-dispatch — no heap.

### Analytic gradient override

For hot-path use, override `signed_distance_grad` analytically. The
cone above:

```rust
# use molrs::types::F;
# use molrs_pack::Region;
# #[derive(Debug, Clone, Copy)]
# pub struct ConeRegion { pub apex: [F; 3], pub axis: [F; 3], pub half_angle_cos: F }
# impl Region for ConeRegion {
#     fn contains(&self, _x: &[F; 3]) -> bool { true }
#     fn signed_distance(&self, _x: &[F; 3]) -> F { 0.0 }
fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
    let dx = x[0] - self.apex[0];
    let dy = x[1] - self.apex[1];
    let dz = x[2] - self.apex[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();
    if r < 1e-12 { return [0.0; 3]; }
    let axis_dot = (dx * self.axis[0] + dy * self.axis[1] + dz * self.axis[2]) / r;
    let inv_r = 1.0 / r;
    // signed_distance = cos(α) - axis_dot ⇒ grad = -∂axis_dot/∂x
    [
        -(self.axis[0] * inv_r - axis_dot * dx * inv_r * inv_r),
        -(self.axis[1] * inv_r - axis_dot * dy * inv_r * inv_r),
        -(self.axis[2] * inv_r - axis_dot * dz * inv_r * inv_r),
    ]
}
# }
```

Then finite-difference check it — same pattern as the `Restraint`
test.

## Custom `Handler`

Goal: a handler that writes a CSV row per step so you can plot the
objective evolution.

```rust,no_run
use std::fs::File;
use std::io::{BufWriter, Write};
use molrs_pack::{F, Handler, PackContext, StepInfo};

pub struct CsvHandler { writer: BufWriter<File> }

impl CsvHandler {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "phase,loop_idx,fdist,frest,improvement_pct")?;
        Ok(Self { writer: w })
    }
}

impl Handler for CsvHandler {
    fn on_step(&mut self, info: &StepInfo, _sys: &PackContext) {
        let _ = writeln!(
            self.writer,
            "{},{},{},{},{}",
            info.phase.phase,
            info.loop_idx,
            info.fdist,
            info.frest,
            info.improvement_pct,
        );
    }
}
```

Handler notes:

- **`on_step` is the only required method.** Everything else has a
  default no-op.
- **`sys` is `&PackContext`, never `&mut`.** Handlers cannot mutate
  packer state — use a `Relaxer` if you need to.
- **Multiple handlers run in registration order.** Register your CSV
  handler before `ProgressHandler` to get a row on every step,
  vice-versa otherwise.
- **`should_stop` is polled every iteration.** Return `true` to break
  the outer loop early. Useful for time budgets or custom convergence
  criteria.

## Custom `Relaxer`

Goal: a relaxer that tries random rigid-body translations and accepts
if the objective decreases.

```rust,no_run
use molrs::types::F;
use molrs_pack::{Relaxer, RelaxerRunner};
use rand::{Rng, RngCore};

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
            let dx = (rng.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) * self.max_delta;
            let dy = (rng.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) * self.max_delta;
            let dz = (rng.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) * self.max_delta;
            let trial: Vec<[F; 3]> = best.iter()
                .map(|p| [p[0] + dx, p[1] + dy, p[2] + dz])
                .collect();
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
```

Relaxer notes:

- **Two-part design.** [`Relaxer`](crate::Relaxer) is the immutable
  builder; [`RelaxerRunner`](crate::RelaxerRunner) holds per-pack
  state. `build()` is called once per target type at `pack()` entry.
- **`evaluate` closure tests trial coords against the full objective**
  without mutating the reference — use it as often as you like.
- **Return `Some(new_coords)` only if something changed.** The packer
  skips unnecessary cache invalidation when you return `None`.
- **`count == 1` required.** Multi-copy targets share reference
  coords; a relaxer that mutates them would silently change all
  copies.

## Testing discipline

| Kind | Location | Convention |
|---|---|---|
| Unit test | `#[cfg(test)] mod tests` in the same file | One `#[test]` fn per behavior |
| Integration test | `tests/<name>.rs` | `use molrs_pack::{…};` only public API |
| Gradient finite-difference | alongside unit test | ε=1e-5, tol=1e-3 |
| Regression vs Packmol | `tests/examples_batch.rs` (`#[ignore]`) | Run with `--ignored --release` |
| Microbench | `benches/<name>.rs` | criterion with `iter_batched` |

Run all:

```bash
cargo test -p molcrafts-molrs-pack --all-features
cargo test -p molcrafts-molrs-pack --release --test examples_batch -- --ignored
```

Rules:

- Every new `Restraint` gets an FD gradient test.
- Every new `Region` gets a boolean-algebra + signed-distance sign
  test and (for hot-path use) an analytic-gradient FD test.
- Every hot-path change gets a microbench following the `fn` +
  `caller` two-bench pattern (see `benches/run_phase.rs`).

## Benchmarking discipline

Catastrophic-regression alarm:

```bash
cargo bench -p molcrafts-molrs-pack --bench pack_end_to_end -- mixture
```

Five workloads available: `mixture` (~2s), `bilayer` (~10s),
`interface` (~5s), `solvprotein` (~30s), `spherical` (~45 min —
18k molecules).

Microbenches (fast, run after any hot-path edit):

```bash
cargo bench -p molcrafts-molrs-pack --bench evaluate_unscaled
cargo bench -p molcrafts-molrs-pack --bench run_iteration
cargo bench -p molcrafts-molrs-pack --bench run_phase
cargo bench -p molcrafts-molrs-pack --bench objective_dispatch
```

Gates (spec §10):

| Scope | Hard | Soft |
|---|---|---|
| Per-fn microbench | ≤ +1% | ≤ 0% |
| Caller microbench (includes indirection) | ≤ +2% | ≤ +1% |
| `pack_end_to_end` (phase-end alarm) | ≤ +10% | ≤ +5% |

Cross the soft gate → attach a flamegraph + one-paragraph root-cause
note. Cross the hard gate → tighten the change or roll back.

**Extract-bench pattern** (hot-path refactors only):

1. `#[cfg(bench)] #[inline(never)] fn F_sentinel(...)` with the
   pre-extraction body.
2. Microbench of the extracted fn + caller.
3. Extract in the same commit.
4. Delete the sentinel at the end of the NEXT refactor cycle
   (not sooner).

## Common pitfalls

- **Gradient sign.** Every `Restraint` accumulates `∂penalty/∂x`.
  Optimizer negates for descent. If your molecules fly out of the
  region, the gradient has the wrong sign — penalty should point
  toward the violation boundary.
- **Rotation convention.** Single-atom tests pass with both LEFT and
  RIGHT Euler multiplication; multi-atom tests don't. Always test
  Euler changes with ≥ 2 atoms.
- **`Cell<f64>` is not `Sync`.** Use `AtomicU64` +
  `f64::to_bits` / `from_bits` for interior mutability in
  `Send + Sync` contexts.
- **1-based atom indexing.** `Target::with_restraint_for_atoms`
  follows Packmol convention. `&[1, 2]` selects the first two atoms.
  Internally converted to 0-based.
- **`count == 1` required for relaxers.** Multi-copy targets share
  reference coords.
- **`radscale` is phase-dependent.** Don't hard-code atomic radii —
  always go through `sys.radius[i]`. `evaluate_unscaled` temporarily
  swaps `radius` with `radius_ini` for user-facing numbers.
- **PBC boxes must be valid.** Zero-length axis returns
  `PackError::InvalidPBCBox`.

## Contributing flow

1. Open `.claude/specs/molrs-pack-plugin-arch.md` — find the closest
   checklist entry. If none, propose a spec edit first.
2. Write a failing test.
3. Implement until it passes.
4. Run the full gate:
   ```bash
   cargo test --all-features
   cargo clippy -- -D warnings
   cargo fmt --all --check
   ```
5. Hot-path changes: run the relevant microbench + `pack_end_to_end`
   before and after. Attach numbers to the PR.
6. Update the spec's Commit log section.
