//! `GyrationTensor::compute` (regression).

use criterion::{Criterion, criterion_group};
use molrs::compute::shape::gyration_tensor::GyrationTensor;
use molrs::compute::traits::Compute;

use crate::helpers;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape/gyration_tensor");
    helpers::configure(&mut group);
    let gyr = GyrationTensor::new();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();
    let deps = helpers::build_deps(&frames, &nlists);

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                gyr.compute(&frames, (&deps.cluster, &deps.centers))
                    .unwrap(),
            );
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
