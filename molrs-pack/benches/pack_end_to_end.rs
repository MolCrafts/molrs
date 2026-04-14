//! Catastrophic-regression alarm for molrs-pack.
//!
//! Reproduces the five canonical Packmol-equivalent workloads checked in under
//! `molrs-pack/examples/pack_*`. Each bench drives `Molpack::pack(...)` with the
//! same parameters the example uses (seed, nloop, molecule counts, constraints),
//! so this bench and `cargo run --example pack_<name>` walk the exact same hot
//! path.
//!
//! Gate: +10% wall-clock regression blocks merge (see `molrs-perf` skill §
//! "Benchmarking during refactors"). This bench is NOT a per-extraction gate —
//! per-function microbenches live in sibling bench files and land alongside
//! their extractions.
//!
//! Usage:
//! ```sh
//! cargo bench -p molrs-pack --bench pack_end_to_end                   # all 5
//! cargo bench -p molrs-pack --bench pack_end_to_end -- pack_mixture   # one
//! cargo bench -p molrs-pack --bench pack_end_to_end -- --test         # smoke
//! ```
//!
//! Runtime expectation (release, single developer machine):
//! `pack_mixture` / `pack_interface` / `pack_bilayer` each ~1–5 min;
//! `pack_solvprotein` ~5–15 min; `pack_spherical` 30–60 min+ (18k molecules ×
//! 800 loops × 10 samples). Filter for targeted work.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::{ExampleCase, Molpack, build_targets, example_dir_from_manifest};

fn bench_case(c: &mut Criterion, case: ExampleCase) {
    let base = example_dir_from_manifest(case);
    let max_loops = case.max_loops();
    let seed = case.seed();

    let mut group = c.benchmark_group("pack_end_to_end");
    group.sample_size(10);
    group.bench_function(case.name(), |b| {
        b.iter_batched(
            || build_targets(case, &base).expect("build targets"),
            |targets| {
                let mut packer = Molpack::new();
                std::hint::black_box(packer.pack(&targets, max_loops, Some(seed)).expect("pack"));
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_mixture(c: &mut Criterion) {
    bench_case(c, ExampleCase::Mixture);
}

fn bench_bilayer(c: &mut Criterion) {
    bench_case(c, ExampleCase::Bilayer);
}

fn bench_interface(c: &mut Criterion) {
    bench_case(c, ExampleCase::Interface);
}

fn bench_solvprotein(c: &mut Criterion) {
    bench_case(c, ExampleCase::Solvprotein);
}

fn bench_spherical(c: &mut Criterion) {
    bench_case(c, ExampleCase::Spherical);
}

criterion_group!(
    benches,
    bench_mixture,
    bench_bilayer,
    bench_interface,
    bench_solvprotein,
    bench_spherical,
);
criterion_main!(benches);
