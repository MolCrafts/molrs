//! Permanent microbench for `molrs_pack::packer::run_phase`.
//!
//! Originally landed in phase A.4.2 alongside a `#[inline(never)]
//! run_phase_sentinel` for the +1% extraction-gate check. The sentinel
//! was deleted at end-of-phase-B per the `molrs-perf` "delete F_sentinel
//! after the next refactor cycle" rule. The fn + caller microbenches
//! below stay in the tree as future-regression guards.
//!
//! Setup: empty-molecule PackContext. With `ntype=0` / `ntotmol=0` the
//! body's handler-loop / comptype-loop / xwork alloc / evaluate /
//! precision short-circuit still execute on empty vectors — this
//! measures **function-call boundary cost** (indirection, inlining) on
//! a trivial body. Full-workload benchmarking lives in
//! `benches/pack_end_to_end.rs` (catastrophic-regression alarm, ≤ +10%).

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::gencan::{GencanParams, GencanWorkspace};
use molrs_pack::handler::Handler;
use molrs_pack::initial::SwapState;
use molrs_pack::movebad::MoveBadConfig;
use molrs_pack::packer::{PhaseOutcome, run_phase};
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
    let mut group = c.benchmark_group("run_phase");
    group.sample_size(50);
    let mb = movebad_cfg();
    let gp = gencan_params();
    group.bench_function("fn", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let out = run_phase(
                    0,
                    0,
                    0,
                    1,
                    10,
                    2.0,
                    0.01,
                    true,
                    &mb,
                    &gp,
                    &mut sys,
                    &mut x,
                    &mut swap,
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

/// Caller microbench: models `pack()`'s `for phase in 0..=ntype` scaffold
/// calling `run_phase` once per phase and matching on the `PhaseOutcome`.
fn bench_caller(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_phase");
    group.sample_size(50);
    let mb = movebad_cfg();
    let gp = gencan_params();

    group.bench_function("caller", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut converged = false;
                for phase in 0..=0usize {
                    let out = run_phase(
                        phase,
                        0,
                        0,
                        1,
                        10,
                        2.0,
                        0.01,
                        true,
                        &mb,
                        &gp,
                        &mut sys,
                        &mut x,
                        &mut swap,
                        &mut runners,
                        &mut handlers,
                        &mut ws,
                        &mut rng,
                    );
                    match out {
                        PhaseOutcome::Continue => {}
                        PhaseOutcome::Converged => {
                            converged = true;
                            break;
                        }
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
