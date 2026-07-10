//! `pmft` category — potentials of mean force and torque (regression).
//!
//! Covers the R12 (r, θ1, θ2) and XY analyzers. Both consume the pooled
//! self-query neighbor lists; R12 also needs per-particle orientations.

use criterion::{Criterion, criterion_group};
use molrs::compute::pmft::{PMFTR12, PMFTR12Args, PMFTXY, PMFTXYArgs};
use molrs::compute::traits::Compute;
use molrs::types::F;

use crate::helpers;

fn bench_r12(c: &mut Criterion) {
    let mut group = c.benchmark_group("pmft/r12");
    helpers::configure(&mut group);
    let pmft = PMFTR12::new(2.0, 8, 6, 6).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();
    let orientations: Vec<Vec<F>> = (0..frames.len())
        .map(|_| vec![0.0_f64; helpers::REG_N])
        .collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                pmft.compute(
                    &frames,
                    PMFTR12Args {
                        nlists: &nlists,
                        orientations: &orientations,
                    },
                )
                .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_xy(c: &mut Criterion) {
    let mut group = c.benchmark_group("pmft/xy");
    helpers::configure(&mut group);
    let pmft = PMFTXY::new(2.0, 2.0, 16, 16).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                pmft.compute(
                    &frames,
                    PMFTXYArgs {
                        nlists: &nlists,
                        query_orientations: None,
                    },
                )
                .unwrap(),
            );
        })
    });
    group.finish();
}

criterion_group!(benches, bench_r12, bench_xy);
