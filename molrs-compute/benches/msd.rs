//! MSD::feed — trajectory-wise MSD accumulation hot path.
//!
//! Feeds M frames of N atoms into a single `MSD` accumulator.

use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs_compute::msd::MSD;

use crate::helpers::{self};

const N_ATOMS: &[usize] = &[500, 2_000, 10_000];
const N_FRAMES: usize = 20;

fn bench_feed(c: &mut Criterion) {
    let mut group = c.benchmark_group("msd/feed");

    for &n in N_ATOMS {
        // Pre-build N_FRAMES worth of frames with different RNG seeds
        // (simulate a trajectory; exact motion unimportant for perf shape).
        let simbox = helpers::pbc_simbox(helpers::BOX_SIZE);
        let frames: Vec<_> = (0..N_FRAMES)
            .map(|t| {
                let pts = helpers::random_positions(n, helpers::BOX_SIZE, 100 + t as u64);
                helpers::frame_from_positions(&pts, simbox.clone())
            })
            .collect();

        let id = BenchmarkId::new(format!("n{n}_frames{N_FRAMES}"), n);
        group.bench_with_input(id, &(), |b, _| {
            b.iter(|| {
                let mut msd = MSD::new();
                for f in &frames {
                    msd.feed(f).expect("msd::feed");
                }
                std::hint::black_box(msd);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_feed);
