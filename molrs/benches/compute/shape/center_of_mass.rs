//! `CenterOfMass::compute` — mass-weighted cluster centers (regression).

use criterion::{Criterion, criterion_group};
use molrs::compute::shape::center_of_mass::CenterOfMass;
use molrs::compute::traits::Compute;

use crate::helpers;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape/center_of_mass");
    helpers::configure(&mut group);
    let com = CenterOfMass::new();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();
    let deps = helpers::build_deps(&frames, &nlists);

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(com.compute(&frames, &deps.cluster).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
