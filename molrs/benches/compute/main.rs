mod center_of_mass;
mod cluster;
mod cluster_centers;
mod dielectric;
mod gyration_tensor;
mod helpers;
mod inertia_tensor;
mod msd;
mod radius_of_gyration;
mod rdf;
mod spectra;
mod van_hove;
#[cfg(feature = "voronoi")]
mod voronoi;

use criterion::criterion_main;

#[cfg(feature = "voronoi")]
criterion_main!(
    rdf::benches,
    msd::benches,
    cluster::benches,
    cluster_centers::benches,
    center_of_mass::benches,
    gyration_tensor::benches,
    inertia_tensor::benches,
    radius_of_gyration::benches,
    dielectric::benches,
    spectra::benches,
    van_hove::benches,
    voronoi::benches,
);

#[cfg(not(feature = "voronoi"))]
criterion_main!(
    rdf::benches,
    msd::benches,
    cluster::benches,
    cluster_centers::benches,
    center_of_mass::benches,
    gyration_tensor::benches,
    inertia_tensor::benches,
    radius_of_gyration::benches,
    dielectric::benches,
    spectra::benches,
    van_hove::benches,
);
