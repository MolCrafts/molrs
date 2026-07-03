//! `dynamics` category — Van Hove correlation G(r, t) (regression).
//!
//! The distinct part `G_d` finds pairs with the same `NeighborQuery` spatial
//! search as the RDF (cutoff = `r_max`), so this is the O(N) regression guard
//! for the self + distinct correlation over a few lags.

use criterion::{Criterion, criterion_group};
use molrs::compute::dynamics::van_hove::VanHove;
use molrs::compute::traits::Compute;

use crate::helpers;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamics/van_hove");
    helpers::configure(&mut group);
    // r_max = neighbour cutoff; lags 0..=2 need >= 3 frames.
    let vh = VanHove::new(helpers::RDF_BINS, helpers::CUTOFF, vec![0, 1, 2]).unwrap();
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(vh.compute(&frames, ()).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
