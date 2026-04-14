//! Per-cluster tensor benches: CenterOfMass, ClusterCenters,
//! GyrationTensor, InertiaTensor, RadiusOfGyration.
//!
//! Shares the same `ClusterResult` built once per N; each tensor is
//! benched over the same clusters.

use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs_compute::center_of_mass::CenterOfMass;
use molrs_compute::cluster::Cluster;
use molrs_compute::cluster_centers::ClusterCenters;
use molrs_compute::gyration_tensor::GyrationTensor;
use molrs_compute::inertia_tensor::InertiaTensor;
use molrs_compute::radius_of_gyration::RadiusOfGyration;
use molrs_compute::traits::Compute;

use crate::helpers::{self, Fixture};

const N_ATOMS: &[usize] = &[500, 2_000, 10_000];

fn bench_tensors(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensors");

    for &n in N_ATOMS {
        let Fixture { frame, nlist, .. } = helpers::fixture(n, 42);
        let cluster_result = Cluster::new(2)
            .compute(&frame, &nlist)
            .expect("cluster setup");

        let centers = ClusterCenters::new();
        let com = CenterOfMass::new();
        let gyr = GyrationTensor::new();
        let inertia = InertiaTensor::new();
        let rg = RadiusOfGyration::new();

        let bench = |g: &mut criterion::BenchmarkGroup<_>, label: &str, body: &dyn Fn()| {
            let id = BenchmarkId::new(format!("{label}_n{n}"), n);
            g.bench_with_input(id, &(), |b, _| {
                b.iter(|| {
                    body();
                });
            });
        };

        bench(&mut group, "cluster_centers", &|| {
            let r = centers
                .compute(&frame, &cluster_result)
                .expect("cluster_centers");
            std::hint::black_box(r);
        });
        bench(&mut group, "center_of_mass", &|| {
            let r = com.compute(&frame, &cluster_result).expect("com");
            std::hint::black_box(r);
        });
        bench(&mut group, "gyration_tensor", &|| {
            let r = gyr.compute(&frame, &cluster_result).expect("gyration");
            std::hint::black_box(r);
        });
        bench(&mut group, "inertia_tensor", &|| {
            let r = inertia.compute(&frame, &cluster_result).expect("inertia");
            std::hint::black_box(r);
        });
        bench(&mut group, "radius_of_gyration", &|| {
            let r = rg.compute(&frame, &cluster_result).expect("rg");
            std::hint::black_box(r);
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tensors);
