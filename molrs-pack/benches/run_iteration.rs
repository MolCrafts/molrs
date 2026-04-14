//! Permanent microbench for `molrs_pack::packer::run_iteration`.
//!
//! Originally landed in phase A.4.3 alongside a `#[inline(never)]
//! run_iteration_sentinel` for the +1% extraction-gate check. The
//! sentinel was deleted at end-of-phase-B per the `molrs-perf` "delete
//! F_sentinel after the next refactor cycle" rule. The fn + caller
//! microbenches below stay as future-regression guards.
//!
//! Setup: empty-molecule PackContext. With `ntotmol=0` the body's
//! pgencan/evaluate/movebad branches run on empty vectors — this
//! measures **function-call boundary cost** (indirection, inlining)
//! on a trivial body. Full-workload benchmarking lives in
//! `benches/pack_end_to_end.rs` (catastrophic-regression alarm, ≤ +10%).

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::gencan::{GencanParams, GencanWorkspace};
use molrs_pack::handler::{Handler, PhaseInfo};
use molrs_pack::initial::SwapState;
use molrs_pack::movebad::MoveBadConfig;
use molrs_pack::packer::{IterOutcome, run_iteration};
use molrs_pack::relaxer::RelaxerRunner;
use molrs_pack::{F, PackContext};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Snapshot = (
    PackContext,
    Vec<F>,
    SwapState,
    GencanWorkspace,
    Vec<(usize, Vec<Box<dyn RelaxerRunner>>)>,
    Vec<Box<dyn Handler>>,
    SmallRng,
);

fn build_snapshot() -> Snapshot {
    let ntotat = 4;
    let mut sys = PackContext::new(ntotat, 0, 0);
    sys.radius.fill(0.75);
    sys.radius_ini.fill(1.5);
    sys.work.radiuswork.resize(ntotat, 0.0);
    let x: Vec<F> = Vec::new();
    let swap = SwapState::init(&x, &sys);
    let ws = GencanWorkspace::new();
    let runners: Vec<(usize, Vec<Box<dyn RelaxerRunner>>)> = Vec::new();
    let handlers: Vec<Box<dyn Handler>> = Vec::new();
    let rng = SmallRng::seed_from_u64(1_234_567);
    (sys, x, swap, ws, runners, handlers, rng)
}

fn phase_info() -> PhaseInfo {
    PhaseInfo {
        phase: 0,
        total_phases: 1,
        molecule_type: None,
    }
}

fn movebad_cfg() -> MoveBadConfig<'static> {
    MoveBadConfig {
        movefrac: 0.05,
        maxmove_per_type: &[],
        movebadrandom: false,
        gencan_maxit: 20,
    }
}

fn gencan_params() -> GencanParams {
    GencanParams::default()
}

fn bench_fn(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_iteration");
    group.sample_size(50);
    let pi = phase_info();
    let mb = movebad_cfg();
    let gp = gencan_params();
    group.bench_function("fn", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut flast = 0.0_f64;
                let mut fimp_prev = F::INFINITY;
                let mut radscale = 1.0_f64;
                let out = run_iteration(
                    0,
                    10,
                    true,
                    0,
                    pi,
                    0.01,
                    true,
                    &mb,
                    &gp,
                    &mut sys,
                    &mut x,
                    &mut swap,
                    &mut flast,
                    &mut fimp_prev,
                    &mut radscale,
                    &mut runners,
                    &mut handlers,
                    &mut ws,
                    &mut rng,
                );
                std::hint::black_box(out);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Caller microbench: one iteration dispatch + `IterOutcome` match,
/// modeling the phase's inner `for loop_idx in 0..max_loops` scaffold.
fn bench_caller(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_iteration");
    group.sample_size(50);
    let pi = phase_info();
    let mb = movebad_cfg();
    let gp = gencan_params();

    group.bench_function("caller", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut flast = 0.0_f64;
                let mut fimp_prev = F::INFINITY;
                let mut radscale = 1.0_f64;
                let mut converged = false;
                let out = run_iteration(
                    0,
                    10,
                    true,
                    0,
                    pi,
                    0.01,
                    true,
                    &mb,
                    &gp,
                    &mut sys,
                    &mut x,
                    &mut swap,
                    &mut flast,
                    &mut fimp_prev,
                    &mut radscale,
                    &mut runners,
                    &mut handlers,
                    &mut ws,
                    &mut rng,
                );
                match out {
                    IterOutcome::Continue => {}
                    IterOutcome::Converged | IterOutcome::EarlyStop => {
                        converged = true;
                    }
                }
                std::hint::black_box(converged);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_fn, bench_caller);
criterion_main!(benches);
