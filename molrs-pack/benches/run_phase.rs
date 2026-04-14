//! Per-extraction microbench for `molrs_pack::packer::run_phase`.
//!
//! Landed in the same commit as the extraction (phase A.4.2) per the
//! `molrs-perf` skill § "Benchmarking during refactors" rule:
//!
//!   > In the same commit as the extraction, land:
//!   > - A criterion microbench of the extracted function.
//!   > - A criterion microbench of the caller.
//!
//! Gates (hard):
//!   - `extracted` ≤ +1% vs. `sentinel`
//!   - `caller_extracted` ≤ +2% vs. `caller_sentinel`
//!
//! Setup: empty-molecule PackContext identical to the unit test in
//! `packer::tests::run_phase_matches_sentinel_on_empty_context`. With
//! `ntype=0` / `ntotmol=0` the body's handler-loop / comptype-loop / xwork
//! allocation / evaluate / precision short-circuit still execute on empty
//! vectors — this is intentional: the gate measures *function-call boundary
//! cost* (indirection, inlining decisions) on a trivial body, which is exactly
//! what the sentinel sibling controls for. A full-workload bench lives in
//! `benches/pack_end_to_end.rs` (catastrophic-regression alarm, ≤ +10%).
//!
//! This bench stays permanently; the sentinel is deleted one Phase A cycle
//! after A.4.2 stabilizes.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::gencan::{GencanParams, GencanWorkspace};
use molrs_pack::handler::Handler;
use molrs_pack::initial::SwapState;
use molrs_pack::movebad::MoveBadConfig;
use molrs_pack::packer::{PhaseOutcome, run_phase, run_phase_sentinel};
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

fn bench_extracted(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_phase");
    group.sample_size(50);
    let mb = movebad_cfg();
    let gp = gencan_params();
    group.bench_function("extracted", |b| {
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

fn bench_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_phase");
    group.sample_size(50);
    let mb = movebad_cfg();
    let gp = gencan_params();
    group.bench_function("sentinel", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let out = run_phase_sentinel(
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
/// calling `run_phase` once per phase and matching on the `PhaseOutcome`
/// variant. The difference between `caller_extracted` and `caller_sentinel`
/// captures any indirection / inlining-boundary cost the function-level bench
/// cannot see.
fn bench_caller(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_phase_caller");
    group.sample_size(50);
    let mb = movebad_cfg();
    let gp = gencan_params();

    group.bench_function("caller_extracted", |b| {
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

    group.bench_function("caller_sentinel", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut converged = false;
                for phase in 0..=0usize {
                    let out = run_phase_sentinel(
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

criterion_group!(benches, bench_extracted, bench_sentinel, bench_caller);
criterion_main!(benches);
