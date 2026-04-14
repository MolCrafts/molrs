//! A.6 dispatch microbench for `Objective::evaluate`.
//!
//! A.6 is a **signature rewire**, not a pure-function extraction: `pgencan`,
//! `gencan`, `tn_ls`, `spg::spgls`, `cg::cg_solve` each swap their
//! `&mut PackContext` argument for `&mut dyn Objective`, and every
//! `sys.evaluate(...)` / `compute_f(...)` / `compute_g(...)` call site
//! becomes `obj.evaluate(...)`. The only new cost at release is the vtable
//! indirection on the `Objective::evaluate` trait method.
//!
//! The extract-bench discipline's sentinel rule targets pure-function
//! extractions (one function → one `F_sentinel`). Duplicating 1000 lines of
//! `gencan` body as a sentinel gives no extra localization over the bench
//! below and costs maintenance. Instead, this bench isolates the **only
//! thing A.6 changes**: the dispatch cost of one `evaluate` call through an
//! inherent impl vs. through a `&mut dyn Objective` receiver.
//!
//! Gates:
//!   - `via_dyn` ≤ +1% vs. `via_inherent` on the same trivial context
//!     (dyn dispatch overhead per call).
//!   - `pack_end_to_end/mixture` ≤ +10% vs. previous commit (catastrophic
//!     alarm; captures any compound effect over ~10k evaluate calls / pack).
//!
//! Setup mirrors `objective_trait_tests::dyn_objective_matches_inherent_evaluate`
//! and the other hot-path extract benches: empty-molecule PackContext
//! (`ntotat=4`, `ntotmol=0`, seeded), `FOnly` mode, empty gradient slot —
//! `evaluate` returns `(0,0,0)` so the bench measures boundary cost, not
//! body cost.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::constraints::EvalMode;
use molrs_pack::objective::Objective;
use molrs_pack::{F, PackContext};

fn build_ctx() -> PackContext {
    let ntotat = 4;
    let mut sys = PackContext::new(ntotat, 0, 0);
    sys.radius.fill(0.75);
    sys.radius_ini.fill(1.5);
    sys.work.radiuswork.resize(ntotat, 0.0);
    sys
}

fn bench_via_inherent(c: &mut Criterion) {
    let mut group = c.benchmark_group("objective_dispatch");
    group.sample_size(50);
    let x: Vec<F> = Vec::new();
    group.bench_function("via_inherent", |b| {
        b.iter_batched(
            build_ctx,
            |mut sys| {
                let out = PackContext::evaluate(&mut sys, &x, EvalMode::FOnly, None);
                std::hint::black_box(out);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_via_dyn(c: &mut Criterion) {
    let mut group = c.benchmark_group("objective_dispatch");
    group.sample_size(50);
    let x: Vec<F> = Vec::new();
    group.bench_function("via_dyn", |b| {
        b.iter_batched(
            build_ctx,
            |mut sys| {
                let obj: &mut dyn Objective = &mut sys;
                let out = obj.evaluate(&x, EvalMode::FOnly, None);
                std::hint::black_box(out);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Caller microbench: models `tn_ls`'s per-iteration evaluate burst
/// (~4 calls — one FOnly, two GradientOnly, one FAndGradient in the typical
/// extrapolation branch). Any dyn-dispatch cost that fails to inline at the
/// call site compounds here.
fn bench_caller_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("objective_dispatch_caller");
    group.sample_size(50);
    let x: Vec<F> = Vec::new();

    group.bench_function("caller_inherent", |b| {
        b.iter_batched(
            build_ctx,
            |mut sys| {
                for _ in 0..4 {
                    let out = PackContext::evaluate(&mut sys, &x, EvalMode::FOnly, None);
                    std::hint::black_box(out);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("caller_dyn", |b| {
        b.iter_batched(
            build_ctx,
            |mut sys| {
                let obj: &mut dyn Objective = &mut sys;
                for _ in 0..4 {
                    let out = obj.evaluate(&x, EvalMode::FOnly, None);
                    std::hint::black_box(out);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_via_inherent,
    bench_via_dyn,
    bench_caller_loop
);
criterion_main!(benches);
