//! `GyrationTensor::compute` across size + frame axes.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use molrs_compute::gyration_tensor::GyrationTensor;
use molrs_compute::traits::Compute;

use crate::helpers;

fn size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("gyration_tensor/size_sweep");
    helpers::configure(&mut group);
    let gyr = GyrationTensor::new();

    for &n in helpers::SIZES {
        let (frames_owned, nlists) = helpers::build_pool(n, helpers::SIZE_SWEEP_FRAMES, 42);
        let frames: Vec<&_> = frames_owned.iter().collect();
        let deps = helpers::build_deps(&frames, &nlists);
        group.throughput(Throughput::Elements(
            (n as u64) * helpers::SIZE_SWEEP_FRAMES as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    gyr.compute(&frames, (&deps.cluster, &deps.centers))
                        .unwrap(),
                );
            })
        });
    }

    group.finish();
}

fn frame_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("gyration_tensor/frame_sweep");
    helpers::configure(&mut group);
    let gyr = GyrationTensor::new();
    let (pool_frames, pool_nlists) =
        helpers::build_pool(helpers::FRAME_SWEEP_N, helpers::MAX_FRAMES, 100);

    for &nf in helpers::FRAME_COUNTS {
        let frames: Vec<&_> = pool_frames.iter().take(nf).collect();
        let nlists: Vec<_> = pool_nlists.iter().take(nf).cloned().collect();
        let deps = helpers::build_deps(&frames, &nlists);
        group.throughput(Throughput::Elements(
            (helpers::FRAME_SWEEP_N as u64) * nf as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    gyr.compute(&frames, (&deps.cluster, &deps.centers))
                        .unwrap(),
                );
            })
        });
    }

    group.finish();
}

criterion_group!(benches, size_sweep, frame_sweep);
