//! `VanHove::compute` — self + distinct Van Hove across the particle-count axis.
//!
//! The distinct part `G_d` now finds pairs with the same `NeighborQuery`
//! spatial search as [`rdf`](molrs::compute::rdf) (cutoff = `r_max`), so this
//! sweep should be ~flat per particle (O(N)) rather than the old O(N²)
//! all-pairs loop. A few frames + small lags so the distinct cross-query
//! dominates the measurement.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use molrs::compute::traits::Compute;
use molrs::compute::van_hove::VanHove;

use crate::helpers;

/// Atom counts for the Van Hove sweep (local, bounded — the distinct part was
/// O(N²) before, so keep the ceiling modest and comparable to the voronoi bench).
const VH_SIZES: &[usize] = &[100, 500, 1_000, 2_000, 4_000];

fn size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("van_hove/size_sweep");
    helpers::configure(&mut group);
    // r_max = neighbour cutoff; lags 0..=2 need ≥3 frames.
    let vh = VanHove::new(helpers::RDF_BINS, helpers::CUTOFF, vec![0, 1, 2]).unwrap();

    for &n in VH_SIZES {
        let (frames_owned, _nlists) = helpers::build_pool(n, 4, 17);
        let frames: Vec<&_> = frames_owned.iter().collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(vh.compute(&frames, ()).unwrap());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, size_sweep);
