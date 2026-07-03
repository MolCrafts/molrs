//! Compute regression benchmarks.
//!
//! One module per `src/compute/` category; each holds ONE small representative
//! input per kernel (regression sizing, see [`helpers`]). The `criterion_main!`
//! list below is grouped by category to mirror the source layout.

mod helpers;

// Structure / trajectory
mod cluster;
mod dynamics;
mod msd;
mod rdf;
mod shape;

// Transport / spectroscopy / dielectric
mod dielectric;
mod spectroscopy;
mod transport;

// Statistical mechanics / structure analysis
mod density;
mod diffraction;
mod distribution;
mod environment;
mod hbond;
mod ml;
mod order;
mod pmft;

#[cfg(feature = "voronoi")]
mod voronoi;

use criterion::criterion_main;

#[cfg(feature = "voronoi")]
criterion_main!(
    // rdf / msd / cluster / shape / dynamics
    rdf::benches,
    msd::benches,
    cluster::benches,
    shape::center_of_mass::benches,
    shape::cluster_centers::benches,
    shape::gyration_tensor::benches,
    shape::inertia_tensor::benches,
    shape::radius_of_gyration::benches,
    dynamics::benches,
    // transport / spectroscopy / dielectric
    transport::benches,
    spectroscopy::benches,
    dielectric::benches,
    // density / diffraction / distribution / environment / hbond / ml / order / pmft
    density::benches,
    diffraction::benches,
    distribution::benches,
    environment::benches,
    hbond::benches,
    ml::benches,
    order::benches,
    pmft::benches,
    // voronoi
    voronoi::benches,
);

#[cfg(not(feature = "voronoi"))]
criterion_main!(
    // rdf / msd / cluster / shape / dynamics
    rdf::benches,
    msd::benches,
    cluster::benches,
    shape::center_of_mass::benches,
    shape::cluster_centers::benches,
    shape::gyration_tensor::benches,
    shape::inertia_tensor::benches,
    shape::radius_of_gyration::benches,
    dynamics::benches,
    // transport / spectroscopy / dielectric
    transport::benches,
    spectroscopy::benches,
    dielectric::benches,
    // density / diffraction / distribution / environment / hbond / ml / order / pmft
    density::benches,
    diffraction::benches,
    distribution::benches,
    environment::benches,
    hbond::benches,
    ml::benches,
    order::benches,
    pmft::benches,
);
