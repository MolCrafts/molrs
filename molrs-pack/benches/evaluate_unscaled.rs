//! Permanent microbench for `molrs_pack::packer::evaluate_unscaled`.
//!
//! Originally landed in phase A.4.1 alongside a `#[inline(never)]
//! evaluate_unscaled_sentinel` for the +1% extraction-gate check. The
//! sentinel was deleted at end-of-phase-B per the `molrs-perf` skill
//! "delete F_sentinel after the next refactor cycle" rule. The fn and
//! caller microbenches below stay in the tree permanently as a
//! future-regression guard.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::packer::evaluate_unscaled;
use molrs_pack::{ExampleCase, Molpack, PackContext, build_targets, example_dir_from_manifest};

/// Build a realistic mid-pack PackContext by warming up `pack_mixture`
/// for one outer iteration, then constructing a lightweight snapshot
/// representative of the hot-path evaluate_unscaled call.
fn build_snapshot() -> (PackContext, Vec<f64>) {
    let targets = build_targets(
        ExampleCase::Mixture,
        &example_dir_from_manifest(ExampleCase::Mixture),
    )
    .expect("build targets");
    let mut packer = Molpack::new();
    let _ = packer.pack(&targets, 1, Some(1_234_567)).expect("pack");

    let ntotat = 400;
    let mut sys = PackContext::new(ntotat, 0, 0);
    sys.radius.iter_mut().for_each(|r| *r = 0.75);
    sys.radius_ini.iter_mut().for_each(|r| *r = 1.5);
    sys.work.radiuswork.resize(ntotat, 0.0);
    (sys, Vec::new())
}

fn bench_fn(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_unscaled");
    group.sample_size(50);
    group.bench_function("fn", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, xwork)| {
                std::hint::black_box(evaluate_unscaled(&mut sys, &xwork));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Caller microbench: models the "post-pgencan statistics" block in `pack()`
/// — one `evaluate_unscaled` call followed by the `fimp` arithmetic, which
/// is the exact sequence the hot path executes per outer-loop iteration.
fn bench_caller(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_unscaled");
    group.sample_size(50);
    group.bench_function("caller", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, xwork)| {
                let (fx, _fdist, _frest) = evaluate_unscaled(&mut sys, &xwork);
                let flast = 1.0_f64;
                let fimp = if flast > 0.0 {
                    -100.0 * (fx - flast) / flast
                } else {
                    0.0
                };
                std::hint::black_box(fimp);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_fn, bench_caller);
criterion_main!(benches);
