//! `diffraction` category — static structure factor S(k) (regression).
//!
//! Covers the Debye (all-pairs sinc) and Direct (reciprocal-lattice) routes.
//! Debye is O(N²) in atom pairs, so this uses a smaller single-frame pool
//! (~300 atoms) than the shared 1k regression pool.

use criterion::{Criterion, criterion_group};
use molrs::compute::diffraction::{StaticStructureFactorDebye, StaticStructureFactorDirect};
use molrs::compute::traits::Compute;

use crate::helpers;

const DIFF_N: usize = 300;

fn bench_debye(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffraction/debye");
    helpers::configure(&mut group);
    let (frames_owned, _) = helpers::build_pool(DIFF_N, 1, 7);
    let frames: Vec<&_> = frames_owned.iter().collect();
    let sk = StaticStructureFactorDebye::linspace(0.5, 8.0, 20).unwrap();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(sk.compute(&frames, ()).unwrap());
        })
    });
    group.finish();
}

fn bench_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffraction/direct");
    helpers::configure(&mut group);
    let (frames_owned, _) = helpers::build_pool(DIFF_N, 1, 7);
    let frames: Vec<&_> = frames_owned.iter().collect();
    // Isotropic radial average needs a periodic box (reciprocal lattice). Keep
    // k_max small — the enumerated k-vector count grows as (k_max·L)³.
    let sk = StaticStructureFactorDirect::isotropic(2.0, 10).unwrap();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(sk.compute(&frames, ()).unwrap());
        })
    });
    group.finish();
}

criterion_group!(benches, bench_debye, bench_direct);
